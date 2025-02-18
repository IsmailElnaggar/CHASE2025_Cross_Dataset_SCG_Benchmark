#imports
import numpy as np
import pandas as pd
from scipy import signal,stats
import pywt


#CWT RELATED HELPER FUNCTIONS 
#These are crude functions just for this paper implementation:
#IEEE/ACM CHASE2025 publication: Cross-Dataset Validation of a Sensor Agnostic Sesimocardiography Peak Detection Method
#no error handling 
#this code makes the assumption that SCG signals will be at 200 Hz
#later revisions will alter functions to consider any reasonable sampling frequency for scg/gcg signals
def normalizesig(data):
    x=np.array(data)
    p1=x-np.min(x)
    p2=np.max(x) - np.min(x)
    return 2 * (p1/p2) - 1

def NormalizeData(data):
    #1 to 0
    data=np.array(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data)) 

def pointmax(x):
    #x is assumed to be a 2d array, like a cwt
    #mean taken across rows, so the scales aka freqs
    
    return np.max(x,axis=0)

def scale_selecter(scales=(5,25,.1),fs=200):
    
    #find scales that correlate to given freq range in freq_scales
    all_scales=np.arange(1.5,400,.001)
    freqz=pywt.scale2frequency("morl",all_scales)/ (1/fs)

    goal_scales=[]

    for i in np.arange(scales[0],scales[1]+1,scales[2]):
        index = np.argmin(np.abs(np.array(freqz)-i))
        goal_scales.append(index)

    true_goal=all_scales[sorted(list(set(goal_scales)))]
    goal_freqz=[]
    for i in true_goal:
        goal_freqz.append(pywt.scale2frequency("morl",i)/ (1/fs))

    freqz=pywt.scale2frequency("morl",true_goal)/ (1/fs)

    scalez=true_goal
    return scalez

def bp_filter(signalz,hp=5,lp=25,fs=200,order=4):
    sos = signal.butter(order,[hp,lp],fs=(fs),btype='bandpass',output='sos')
    filtsig=signalz-np.mean(signalz)
    filts=signal.sosfiltfilt(sos,filtsig,axis=0)
    filts=filts/np.std(filts)
    
    return filts

def cwt_waveform_maker(signalz,scalez,wavelet_name="morl",fs=200,abs_val=True):
    dt=1/fs
    
    Zxx, f = pywt.cwt(signalz, scalez, wavelet_name,dt)
    
    if abs_val==True:
        Zxx=np.abs(Zxx)
    else:
        pass
    
    return Zxx,f

def cwt_stat_waveform(Zxx,stat="meanRecs",smoothwindow=10):
            
    if stat == "meanRecs":
        avg_v=np.zeros(Zxx.shape[1])
        avg_v[0:smoothwindow]=np.mean(Zxx[:,0:smoothwindow],axis=0)
        avg_v[-smoothwindow:]=np.mean(Zxx[:,-smoothwindow:],axis=0)
        for i in range(smoothwindow,Zxx.shape[1]-smoothwindow,):
            
            avg_v[i]=np.mean(Zxx[:,i-(smoothwindow):i+(smoothwindow)]) 
            
    else:
        avg_v=np.max(Zxx,axis=0)
        
    N=smoothwindow*2
    avg_v_meanz=rollingmax(avg_v,N)
    N=smoothwindow*4
    avg_v_mean=rollingmean(avg_v_meanz,N)
        
    return avg_v_mean
    
def fp_ampd(x_array, scale):
    """  
    Find peaks in quasi-periodic noisy signals using AMPD algorithm.
	edited and pulled from: https://github.com/ig248/pyampd/blob/master/pyampd
    Extended implementation handles peaks near start/end of the signal. #removed
    Optimized implementation by Igor Gotlibovych, 2018

    Parameters
    ----------
    x_array : ndarray
        1-D array on which to find peaks
    scale : int
        specify maximum scale window size of (2 * scale + 1)


    Returns
    -------
    pks: ndarray
        The ordered array of peak indices found in `x`

    """

    L = int(scale * 2)
    xL = len(x_array)
    my_maximum = np.zeros([L, xL], dtype=bool)

    for k in range(1, L):
        my_maximum[k - 1, k : xL - k] = (
            x_array[k : xL - k] > x_array[0 : xL - 2 * k]
        ) & (x_array[k : xL - k] > x_array[2 * k : xL])

    #scale with largest max
    s_max = np.sum(my_maximum, axis=1)
    m_scale = np.argmax(s_max)
    pks_select = np.min(my_maximum[0:m_scale, :], axis=0)
    pks = np.flatnonzero(pks_select)

    return pks


    
def twod_conv_symmetric(my_cwt, xk, yk):
    #check kernel sizes are odd
    if xk % 2 == 0:
        xk += 1
    if yk % 2 == 0:
        yk += 1

    #COLUMNS
    kernel_size = yk
    linear_kernel = signal.windows.triang(kernel_size)
    linear_kernel /= linear_kernel.sum()
    my_cwt = np.apply_along_axis(
        lambda m: np.convolve(m, linear_kernel, mode='same'),
        axis=0, arr=my_cwt)

    #ROWS
    kernel_size = xk
    linear_kernel = signal.windows.triang(kernel_size)
    linear_kernel /= linear_kernel.sum()
    my_cwt = np.apply_along_axis(
        lambda m: np.convolve(m, linear_kernel, mode='same'),
        axis=1, arr=my_cwt)

    return my_cwt

def ismail_peak_detector(inputsignal,fs,scales):
    
    #cwt creation
    my_cwt,f = cwt_waveform_maker(inputsignal, scales,wavelet_name="morl",fs=fs,abs_val=True)
    
    #2d convolution application
    my_cwt = twod_conv_symmetric(my_cwt, 75, 65) # best version
    
    #stepped cwt thresholding
    my_cwt = cwt_step_thresholder(my_cwt, fs, 0.5, 0.45, 200,3,0.9) #.5 seconds
    my_cwt = cwt_step_thresholder(my_cwt, fs, 0.35, 0.2625, 100,3,0.9) # .35 seconds
    
    #1d signal creation from cwt
    my_stat_wav=normalizesig(pointmax(np.abs(my_cwt)))
    xt= signal.windows.triang(66) # ~300 ms at 200 hz
    my_stat_wav= np.apply_along_axis(lambda m : signal.convolve(m,xt,mode="same"),axis=0,arr=my_stat_wav)
    xt= signal.windows.triang(22) # ~100 ms at 200 hz
    my_stat_wav= np.apply_along_axis(lambda m : signal.convolve(m,xt,mode="same"),axis=0,arr=my_stat_wav)
    my_stat_wav=normalizesig(my_stat_wav)

    env=my_stat_wav
    plocs=fp_ampd(env, fs)

    return env,plocs

def cwt_step_thresholder(my_cwt,fs,step_duration,overlap_duration,cwt_split,hf=3,lf=0.5):
    
    cwt_len=my_cwt.shape[1]
    step_samples = int(step_duration * fs) 
    overlap_samples = int(overlap_duration * fs)
    
    for s in range(0, cwt_len - step_samples+1, step_samples - overlap_samples):
    
        e=s+step_samples
        cwt_threshold_hf=stats.iqr(my_cwt[0:cwt_split,s:e]) * hf
        cwt_threshold_lf=np.abs(my_cwt[cwt_split:,s:e]).max() * lf
        
        my_cwt[0:cwt_split,s:e] = np.where(np.abs(my_cwt[0:cwt_split,s:e])>= cwt_threshold_hf,my_cwt[0:cwt_split,s:e],0)
        my_cwt[cwt_split:,s:e] = np.where(np.abs(my_cwt[cwt_split:,s:e])>= cwt_threshold_lf,my_cwt[cwt_split:,s:e],0)

    return my_cwt   

def split_tuple_list_np(tuple_list):
    array = np.array(tuple_list)
    list_A = array[:, 0].tolist()
    list_B = array[:, 1].tolist()
    return list_A, list_B

def find_values_between(list_A, list_B):
    """
    result i is the index in the list of tuples, the start and end of a peak slope
    result j is the index of the peak locations, the max points in the peaks 
    
    """
    result = []
    for i, (a_start, a_end) in enumerate(list_A):
        for j, b in enumerate(list_B):
            if a_start < b < a_end:
                result.append((i, j))
    return result

def detect_peaks_numpy(inputsignal,  window_size, std_multiplier=0.1, threshold_q=1):
    
    timestamps=np.arange(len(inputsignal))
    filtered = np.zeros_like(inputsignal, dtype=bool)
    ma = np.convolve(inputsignal, np.ones(window_size)/window_size, mode='same')
    std = pd.Series(inputsignal).rolling(window=window_size, min_periods=1).std().to_numpy()
    threshold = ma + std_multiplier * std

    while True:
        filtered = inputsignal >= threshold
        if filtered.mean() < threshold_q:
            break
        threshold = ma + std_multiplier * (std[~filtered].mean())

    return timestamps[filtered], filtered