import glob
import numpy as np
import pandas as pd
from scipy import signal

def data_parser(data_path,sampling_freq,ecg_key,scg_key):
    """
    ###
    Input:
       data_path : string, path to a given dataset which will be downsampled to 200 Hz
       sampling_freq: int, int value of original sampling frequency used in a given dataset
       ecg_key: string, key name of ecg channel you would like to select
       scg_key: string, key name of ecg channel you would like to select
    ###

    ###
    Output:
        Data : Dictionary, dict containing resampled data.
        Dictionary will be nested with the top layer containing patient IDs
        Each patient ID will contain an ECG key and a SCG Z-axis Key
    ###    
    """

    #init data_dict
    data_dict=dict()
    
    #need if statement here because VHD dataset contains two different sampling frequencies
    #for other dataset approach can be more streamlined
    if "Formatted_VHD_Data" in data_path:
        formatted_data=glob.glob(data_path+"/*")

        for k in formatted_data:
            mykey=k.split("\\")[-1]
            

            if mykey.startswith("CP-") or mykey.startswith("UP-") and int(mykey.split("-")[1]) <= 21:
                sampling_freq = 256
                
            elif mykey.startswith("UP-") and 22 <= int(mykey.split("-")[1]) <= 30:
                sampling_freq = 512
        
            data_dict[mykey]={key: 0  for key in [ecg_key,scg_key]}
            
            data=pd.read_csv(k)
            print("RUNNING FOR: ",mykey,sampling_freq)
            for col in [ecg_key,scg_key]:
    
                x=data[col].values
                xmr=x-np.mean(x)
                #low pass filter before downsampling to 200hz
                sos = signal.butter(4,[100],fs=(sampling_freq),btype='lowpass',output='sos')
                bpfilt= signal.sosfiltfilt(sos,xmr,axis=0)
                bpfilt=bpfilt/np.std(bpfilt)
                data_dict[mykey][col]=signal.resample(bpfilt,int((len(bpfilt)/sampling_freq)*200))

    else:
        formatted_data=glob.glob(data_path+"/*")
        for k in formatted_data:
            mykey=k.split("\\")[-1]
        
            data_dict[mykey]={key: 0  for key in [ecg_key,scg_key]}
    
            data=pd.read_csv(k)
            print("RUNNING FOR: ",mykey)
            for col in [ecg_key,scg_key]:
    
                x=data[col].values
                xmr=x-np.mean(x)
                #low pass filter before downsampling to 200hz
                sos = signal.butter(4,[100],fs=(sampling_freq),btype='lowpass',output='sos')
                bpfilt= signal.sosfiltfilt(sos,xmr,axis=0)
                bpfilt=bpfilt/np.std(bpfilt)
                data_dict[mykey][col]=signal.resample(bpfilt,int((len(bpfilt)/sampling_freq)*200))        

    return data_dict

        
