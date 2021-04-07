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
    


def plot_sensor_data(data_array:np.ndarray, title:str, ylabels:str):
    
    N_subplots = int(data_array.shape[1] - 1)
    assert len(ylabels) == N_subplots
    
    t = data_array[:,0] * 1e-3
    
    fig = plt.figure()
    
    for k in range(N_subplots):
        plt.subplot(N_subplots,1,k+1)
        plt.plot(t, data_array[:,k + 1])
        plt.xlabel("Time in s")
        plt.ylabel(ylabels[k])
        plt.grid("major")
        plt.grid(which="minor", linestyle="--")
        plt.minorticks_on()
        
        if k == 0:
            plt.title(title)

    
    plt.tight_layout()
    plt.show()
    return fig