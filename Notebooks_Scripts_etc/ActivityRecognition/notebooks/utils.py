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

        


