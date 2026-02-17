import numpy as np
from scipy import linalg,fftpack


# CURRENT SVMD FUNCTIONS USED TO CREATE FIRST SET OF RESULTS ACROSS ALL DATASETS
def waveform_factor(imf):
    """
    Compute the waveform factor of an IMF.

    Parameters:
    - imf: 1D numpy array, the IMF.

    Returns:
    - wf: Waveform factor.
    """
    numerator = np.sqrt(np.mean(imf**2))
    denominator = np.mean(np.abs(imf))
    return numerator / denominator

def reconstruct_signal(modes, threshold):#theshold set to 1 in current implementation
    """
    Reconstruct the AO signal from selected modes based on waveform factor.

    Parameters:
    - modes: List of IMFs.
    - threshold: Average waveform factor threshold.

    Returns:
    - reconstructed_signal: Reconstructed AO signal.
    """
    wf_values = [waveform_factor(imf) for imf in modes]
    avg_wf = np.mean(wf_values)

    selected_modes = [modes[i] for i, wf in enumerate(wf_values) if wf > avg_wf * threshold]
    reconstructed_signal = np.sum(selected_modes, axis=0)
    
    return reconstructed_signal

def svmd(signal, alpha, tol, max_iter=500, max_modes=10):
    """
    Successive Variational Mode Decomposition (SVMD) with fixed mode cap.

    Parameters:
    - signal: 1D numpy array, input signal.
    - alpha: Regularization parameter (balancing spectral smoothness).
    - tol: Convergence tolerance.
    - max_iter: Maximum number of iterations for each mode.
    - max_modes: Maximum number of modes to extract.

    Returns:
    - modes: List of IMFs (Intrinsic Mode Functions).
    """
    signal_length = len(signal)
    residual = signal.copy()
    modes = []

    for mode_idx in range(max_modes):  #range max_modes
        #int mode and lambda_k (cent freq)
        mode = np.zeros(signal_length)
        lambda_k = np.zeros(signal_length)
        step = 1.0

        for iteration in range(max_iter):
            #update mode using FFT-based wiener filtering
            residual_fft = fftpack.fft(residual - mode + lambda_k / 2)
            mode_fft = residual_fft / (1 + alpha * (2 * np.pi * np.fft.fftfreq(signal_length))**2)
            #update mode in the timedomain
            new_mode = np.real(fftpack.ifft(mode_fft))

            #conv check break
            if np.linalg.norm(new_mode - mode) / np.linalg.norm(mode) < tol:
                break

            mode = new_mode
            #update multiplier
            lambda_k = lambda_k + step * (residual - mode)

        #suubtract extracted mode from the residual
        residual -= mode
        modes.append(mode)

        #break if residual energy is below set tolerance
        if np.linalg.norm(residual) / np.linalg.norm(signal) < tol:
            break

    return modes