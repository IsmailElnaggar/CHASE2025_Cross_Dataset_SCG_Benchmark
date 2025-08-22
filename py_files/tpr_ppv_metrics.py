import numpy as np
from scipy.interpolate import interp1d
#interpolate nans values for hr window calc
def interpolate_nans(arr):
    nans = np.isnan(arr)
    valid_indices = np.where(~nans)[0]
    if len(valid_indices) > 1:  #ensure there are two valid points for interpolation
        interp_func = interp1d(valid_indices, arr[valid_indices], kind="linear", fill_value="extrapolate")

        arr[nans] = interp_func(np.where(nans)[0])
    return arr


def calculate_metricsv8(P, R, delta=50):
    """
    Calculates sensitivity (TPR), precision (PPV), and RMSE for beat detection.
    
    Matching procedure for TPR/PPV:
      For each reference beat r_i:
        - Define the True Positive (TP) window: [r_i - delta, r_i + delta].
        - For r_i (if i < N-1), also define a False Positive (FP) window: (r_i + delta, r_{i+1}).
        - For the last reference beat, only the TP window is used.
        - In the TP window, if one or more detected peaks exist, the one closest to r_i
          is taken as a true positive and recorded; extra detections in that window count as FP.
        - Detections in the FP window are normally counted as FP. However, if a candidate in the FP window
          for beat r_i lies in the “look‐ahead zone” [r_{i+1} - delta, r_{i+1}) (and if r_i is not one of the last two beats),
          then we do not count it as an FP for r_i so that it can serve as a candidate for r_{i+1}.
        - If no detection is found in the TP window, the reference beat is counted as a FN.
      Unmatched detections are also counted as FP.
    
    RMSE Calculation:
      For each reference beat r_i (i = 0 ... N-1), we build a candidate value for RMSE:
        - If a true positive was found (stored in p_matched), we use that.
        - Otherwise, for i < N-1, we search the extended window [r_i - delta, r_{i+1}]
          for any detected peak and choose the one closest to r_i.
        - If no candidate is found, the candidate remains NaN.
      Then, for each interval [r_i, r_{i+1}] (i = 0...N-2), if both candidate values are available,
      we compute:
          error = ( (r_{i+1} - r_i) - (candidate[i+1] - candidate[i]) )
      and RMSE is computed as the root-mean-square of these errors.
    
    Additional rules:
      - If len(R) < 10 or len(P) < 10, metrics (TPR, PPV, RMSE) are set to np.nan. # debugging rule 
    
    Parameters:
      P (list or np.array): Detected peak positions.
      R (list or np.array): Reference (ground truth) peak positions.
      delta (int or float): Matching threshold (in same units as P and R). Must be greater than 0.
    
    Returns:
      tpr (float): Sensitivity (TPR) in percentage.
      ppv (float): Precision (PPV) in percentage.
      rmse (float): RMSE computed on beat-to-beat intervals.
      TP (int): Number of true positives.
      FP (int): Number of false positives.
      FN (int): Number of false negatives.
      P (np.array): Detected peak positions. # for debugging
      R (np.array): Reference (ground truth) peak positions. # for debugging 
    """
    
    #convert inputs to numpy arrays.
    P = np.array(P)
    R = np.array(R)
    
    if len(R) < 2 or len(P) < 2:
        return np.nan, np.nan, np.nan, 0, 0, 0

    TP = 0
    FP = 0
    FN = 0

    #used: boolean array indicating if a predicted peak has been assigned.
    used = np.zeros(len(P), dtype=bool)
    
    #p_matched stores the detected peak chosen as a true positive for each reference beat.
    #we record one value per reference beat (including the last one).
    p_matched = np.empty(len(R))
    p_matched.fill(np.nan)
    
    #--- matching for TPR/PPV ---
    #process beats 0 to N-2.
    for i in range(len(R) - 1):
        r = R[i]
        r_next = R[i+1]
        
        #define TP window: [r - delta, r + delta]
        tp_mask = (P >= (r - delta)) & (P <= (r + delta)) & (~used)
        tp_indices = np.where(tp_mask)[0]
        
        #define FP window: (r + delta, r_next)
        fp_mask = (P > (r + delta)) & (P < r_next) & (~used)
        fp_indices = np.where(fp_mask)[0]
        
        #process TP window.
        if tp_indices.size > 0:
            distances = np.abs(P[tp_indices] - r)
            best_idx = tp_indices[np.argmin(distances)]
            TP += 1
            used[best_idx] = True
            p_matched[i] = P[best_idx]
            
            #any extra detections in TP window count as FP.
            extra_tp = np.delete(tp_indices, np.argmin(distances))
            if extra_tp.size > 0:
                FP += len(extra_tp)
                used[extra_tp] = True
        else:
            FN += 1
        
        #process FP window with look-ahead.
        if fp_indices.size > 0:
            #for beats i where i < (N-2), look ahead.
            #for the penultimate beat (i == N-2), we process FP normally.
            if i < len(R) - 2:
                for idx in fp_indices:
                    p_val = P[idx]
                    #look-ahead zone: [r_next - delta, r_next)
                    if p_val >= (r_next - delta):
                        #do not mark this as FP now; leave it unassigned for r_next.
                        continue
                    else:
                        FP += 1
                        used[idx] = True
            else:
                #for the penultimate beat, process FP normally.
                FP += len(fp_indices)
                used[fp_indices] = True

    #process the last reference beat separately.
    r_last = R[-1]
    tp_mask_last = (P >= (r_last - delta)) & (P <= (r_last + delta)) & (~used)
    tp_indices_last = np.where(tp_mask_last)[0]
    if tp_indices_last.size > 0:
        distances = np.abs(P[tp_indices_last] - r_last)
        best_idx = tp_indices_last[np.argmin(distances)]
        TP += 1
        used[best_idx] = True
        p_matched[-1] = P[best_idx]
        extra_tp = np.delete(tp_indices_last, np.argmin(distances))
        if extra_tp.size > 0:
            FP += len(extra_tp)
            used[extra_tp] = True
    else:
        FN += 1

    #any remaining unassigned detections are counted as FP.
    remaining = np.where(~used)[0]
    if remaining.size > 0:
        FP += len(remaining)
        used[remaining] = True

    #--- RMSE calculation ---
    #build candidate array for RMSE.
    #for each beat, if p_matched is available, use it.
    #otherwise, for beats 0 to N-2, search the extended window [r - delta, r_next] for a candidate.
    p_candidate = np.copy(p_matched)
    for i in range(len(R) - 1):
        if np.isnan(p_candidate[i]):
            r = R[i]
            r_next = R[i+1]
            ext_mask = (P >= (r - delta)) & (P < r_next)
            ext_indices = np.where(ext_mask)[0]
            if ext_indices.size > 0:
                distances = np.abs(P[ext_indices] - r)
                best_idx = ext_indices[np.argmin(distances)]
                p_candidate[i] = P[best_idx]
    #for the last beat, if no candidate, search its TP window.
    if np.isnan(p_candidate[-1]):
        r_last = R[-1]
        ext_mask = (P >= (r_last - delta)) & (P <= (r_last + delta))
        ext_indices = np.where(ext_mask)[0]
        if ext_indices.size > 0:
            distances = np.abs(P[ext_indices] - r_last)
            best_idx = ext_indices[np.argmin(distances)]
            p_candidate[-1] = P[best_idx]
    
    #compute RMSE using intervals [r_i, r_{i+1}] for which both candidate values exist.
    errors = []
    for i in range(len(R) - 1):
        if np.isnan(p_candidate[i]) or np.isnan(p_candidate[i+1]):
            continue
        rr_interval = R[i+1] - R[i]
        pp_interval = p_candidate[i+1] - p_candidate[i]
        errors.append(rr_interval - pp_interval)
    errors = np.array(errors)
    if errors.size > 0:
        rmse = np.sqrt(np.mean(errors ** 2))
    else:
        rmse = np.nan

    #compute TPR and PPV.
    tpr = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else np.nan
    ppv = (TP / (TP + FP)) * 100 if (TP + FP) > 0 else np.nan
    
    return tpr, ppv, rmse, TP, FP, FN, P, R

#example usage:
if __name__ == "__main__":
    #example detected peaks and reference peaks.
    P_example = [100, 130, 160, 250, 300, 330, 360, 450, 500, 530, 560, 660, 710]
    R_example = [105, 255, 405, 555, 705, 855, 1005, 1155, 1305, 1455, 1605, 1755]
    
    tpr, ppv, rmse, TP, FP, FN, P, R = calculate_metricsv8(P_example, R_example, delta=30)
    print("TPR: {:.2f}%\nPPV: {:.2f}%\nRMSE: {}\nTP: {}\nFP: {}\nFN: {}"
          .format(tpr, ppv, rmse, TP, FP, FN))
