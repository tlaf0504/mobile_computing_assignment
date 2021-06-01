import os
import glob
import numpy as np
import scipy.signal as sp 
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from typing import Union

def resample_3d_timeseries(y:np.ndarray, t:np.ndarray, f_resample:Union[float,int]=1e3):
    t_mod = t - t[0]
    
    t_resample = np.arange(start=0.0, stop=t_mod[-1], step=1/f_resample)
    f1_resample = interp.interp1d(t_mod, y[:,0])
    f2_resample = interp.interp1d(t_mod, y[:,1])
    f3_resample = interp.interp1d(t_mod, y[:,2])

    return t_resample, np.column_stack((f1_resample(t_resample), f2_resample(t_resample), f3_resample(t_resample)))
    


def plot_sensor_data(data_array:np.ndarray, title:str, ylabels:str, fig=None, curve_label=None):
    
    N_subplots = int(data_array.shape[1] - 1)
    assert len(ylabels) == N_subplots
    
    t = data_array[:,0]
    if fig is None:
        fig_ = plt.figure(figsize=(16,9))
        
    else:
        fig_ = fig
    
    for k in range(N_subplots):
        if fig is None:
            ax = fig_.add_subplot(N_subplots,1,k+1)
        else:
            ax = fig_.axes[k]

        if curve_label is None:
            ax.plot(t, data_array[:,k + 1])
        else:
            ax.plot(t, data_array[:, k+1], label=curve_label)

        ax.set_xlabel("Time in s")
        ax.set_ylabel(ylabels[k])
        ax.grid("major")
        ax.grid(which="minor", linestyle="--")
        ax.legend()
        plt.minorticks_on()
        
        if k == 0:
            ax.set_title(title)

    
    plt.tight_layout()
    plt.show()
    return fig_

def load_data(path:str, src_type="txt"):
    if src_type == "txt":
        data = np.loadtxt(path)
    elif src_type == "npy":
        data = np.load(path, allow_pickle=True)
    else:
        raise Exception("File-type {:s} not implemented.".format(src_type))
    
    return data

def clip_data(t, signals, T_start, T_end):
    idx_clip_start = np.where(t >= T_start)[0][0]
    idx_clip_end = np.where(t <= T_end)[0][-1]

    return idx_clip_start, idx_clip_end, t[idx_clip_start:idx_clip_end + 1], signals[idx_clip_start:idx_clip_end + 1, :]


def separate_and_save_testset_timeframes(t:np.ndarray, signals:np.ndarray, frame_width:float, filename_template:str):
    t_start = t[0]
    t_end = t[-1]

    t_frame_start = t_start
    t_frame_end = t_frame_start + frame_width
    frame_idx = 0
    
    while t_frame_end <= t_end:
        idx_start = np.where(t >= t_frame_start)[0][0]
        idx_end = np.where(t <= t_frame_end)[0][-1]

        frame = signals[idx_start:idx_end+1, :]
        t_frame = t[idx_start:idx_end+1]

        filename = filename_template + f"{frame_idx}.csv"
        print(f"Writing file to {filename}...", end="")
        np.savetxt(filename, np.column_stack((t_frame, frame)), fmt="%e", delimiter=";")
        print("Done")

        t_frame_start = t_frame_end
        t_frame_end = t_frame_start + frame_width
        frame_idx = frame_idx + 1


def load_split_and_store_activity_timeframes(
    data_source_directory: str, 
    data_destination_directory: str,
    T_frame: float,
    T_overlap: float,
    gyro_filename_pattern: str = "*gyro*_resampled_clipped.npy"):

    clipped_and_resampled_gyro_data_files = glob.glob(os.path.join(data_source_directory, gyro_filename_pattern))
    N_files = len(clipped_and_resampled_gyro_data_files)

    print("Found {:d} gyro-files in directory {:s}.".format(N_files, data_source_directory))

    for k in range(N_files):
        gyro_file = clipped_and_resampled_gyro_data_files[k]

        # Extract the timeframes for one activity-sample
        __load_split_and_store_activity_timeframes_single(
            gyro_file,
            data_destination_directory,
            T_frame,
            T_overlap)


def __load_split_and_store_activity_timeframes_single(
    gyro_file: str, 
    data_destination_directory: str,
    T_frame: float,
    T_overlap: float):

    accel_file = gyro_file.replace("gyro", "accel")

    # Load data
    data_gyro = np.load(gyro_file, allow_pickle=True)
    data_accel = np.load(accel_file, allow_pickle=True)


    # Derive sampling-frequency
    fs = np.round(1 / (data_gyro[1,0] - data_gyro[0,0]))

    # Number of samples in one timeframe
    ns_frame = int(np.floor(T_frame * fs))

    # Number of samples in the overlapping area
    ns_overlap = int(np.floor(T_overlap * fs))

    # Number of samples in the non-overlapping area
    ns_non_overlap = int(ns_frame - ns_overlap)

    # Number of samples in the gyro- and accerelometer data-arrays. The number of rows of both arrays are forced to be equal in the pre-processing
    N_data_samples = data_gyro.shape[0]
    
    # The total number of timeframes within the current time-series
    N_frames = int(np.floor(1 + (N_data_samples - ns_frame) / float(ns_non_overlap)))

    # Destination directory is not cleaned up by the script. Do this by hand if wanted.
    if not os.path.isdir(data_destination_directory):
        print("Creating directory {:s}".format(data_destination_directory))
        os.mkdir(data_destination_directory)
        
    # The the filenames for gyro- and accelerometer files without the ".npy" suffix
    filename_gyro_no_suffix = os.path.basename(gyro_file)[:-4]
    filename_accel_no_suffix = os.path.basename(accel_file)[:-4]
        
    for k in range(N_frames):
        idx_start = k * ns_non_overlap
        idx_end = idx_start + ns_frame - 1
        
        timeframe_gyro = data_gyro[idx_start:idx_end+1, :]
        timeframe_accel = data_accel[idx_start:idx_end+1, :]
        
        filename_gyro_timeframe_k = filename_gyro_no_suffix + "_frame_{:d}.npy".format(k)
        filename_accel_timeframe_k = filename_accel_no_suffix + "_frame_{:d}.npy".format(k)
        
        np.save(
            os.path.join(data_destination_directory, filename_gyro_timeframe_k),
            timeframe_gyro,
            allow_pickle=True
        )
        
        np.save(
            os.path.join(data_destination_directory, filename_accel_timeframe_k),
            timeframe_accel,
            allow_pickle=True
        )










