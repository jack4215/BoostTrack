import glob
import os
import numpy as np
import shutil

# Try to import gaussian_filter1d, provide a numpy fallback if missing
try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Using simple moving average fallback for smoothing.")

def gaussian_filter1d_numpy(input_array, sigma):
    """
    Simple fallback for gaussian smoothing using numpy convolution.
    """
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(input_array, kernel, mode='same')

def write_results_no_score(filename, results):
    """Writes results in MOT style to filename."""
    save_format = "{frame},{id},{x1},{y1},{w},{h},{c},-1,-1,-1\n"
    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids, conf in results:
            for tlwh, track_id, c in zip(tlwhs, track_ids, conf):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                    c=round(c, 2)
                )
                f.write(line)


def filter_targets(online_targets, aspect_ratio_thresh, min_box_area):
    """Removes targets not meeting threshold criteria."""
    online_tlwhs = []
    online_ids = []
    online_conf = []
    for t in online_targets:
        tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
        tid = t[4]
        tc = t[5]
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_conf.append(tc)
    return online_tlwhs, online_ids, online_conf


def dti(txt_path, save_path, n_min=25, n_dti=20):
    def dti_write_results(filename, results):
        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
        with open(filename, "w") as f:
            for i in range(results.shape[0]):
                frame_data = results[i]
                frame_id = int(frame_data[0])
                track_id = int(frame_data[1])
                x1, y1, w, h = frame_data[2:6]
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
                f.write(line)

    seq_txts = sorted(glob.glob(os.path.join(txt_path, "*.txt")))
    
    for seq_txt in seq_txts:
        seq_name = seq_txt.replace("\\", "/").split("/")[-1]
        print(f"Processing: {seq_name}")
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
        if len(seq_data) == 0:
            continue
            
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        
        for track_id in range(min_id, max_id + 1):
            index = seq_data[:, 1] == track_id
            tracklet = seq_data[index]
            
            if tracklet.shape[0] == 0:
                continue
            
            # --- 1. Linear Interpolation (Original DTI) ---
            tracklet_dti = tracklet
            n_frame = tracklet.shape[0]
            
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / (
                                right_frame - left_frame
                            ) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1] # interpolated frames get confidence 1 (or -1 to mark them)
                    tracklet_dti = np.vstack((tracklet, data_dti))
            
            # Sort by frame index before smoothing
            tracklet_dti = tracklet_dti[tracklet_dti[:, 0].argsort()]

            # --- 2. [Optimization]: Gaussian Smoothing (GSI) ---
            # We apply smoothing to x, y, w, h components
            # Sigma = 1.0 is conservative but effective for jitter removal
            sigma = 1.0 
            if tracklet_dti.shape[0] > 5: # Only smooth if track is long enough
                if HAS_SCIPY:
                    tracklet_dti[:, 2] = gaussian_filter1d(tracklet_dti[:, 2], sigma) # x
                    tracklet_dti[:, 3] = gaussian_filter1d(tracklet_dti[:, 3], sigma) # y
                    tracklet_dti[:, 4] = gaussian_filter1d(tracklet_dti[:, 4], sigma) # w
                    tracklet_dti[:, 5] = gaussian_filter1d(tracklet_dti[:, 5], sigma) # h
                else:
                    tracklet_dti[:, 2] = gaussian_filter1d_numpy(tracklet_dti[:, 2], sigma)
                    tracklet_dti[:, 3] = gaussian_filter1d_numpy(tracklet_dti[:, 3], sigma)
                    tracklet_dti[:, 4] = gaussian_filter1d_numpy(tracklet_dti[:, 4], sigma)
                    tracklet_dti[:, 5] = gaussian_filter1d_numpy(tracklet_dti[:, 5], sigma)

            seq_results = np.vstack((seq_results, tracklet_dti))
            
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        dti_write_results(save_seq_txt, seq_results)