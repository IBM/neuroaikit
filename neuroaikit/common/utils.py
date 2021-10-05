import numpy as np
import pickle
import os
import copy


def override_config(config, kwargs):
    """ Helper function to override the 'config' with the options present in 'kwargs'. """
    config = copy.deepcopy(config)
    for k, v in kwargs.items():
        if k not in config.keys():
            print('WARNING: Overriding non-existent kwargs key', k)
        config[k] = v
    return config


def transform_rate(data2D, N_ts, max_is_present_for, seed=0):
    """
        Transforms input data into spike trains encoding values using rate-coding.
        N_ts - number of timesteps to generate (length of the spike trains)
        max_is_present_for - expected number of spikes for the maximum value of 1.0
        seed - for reproducibility
    """
    np.random.seed(seed)
    data = []
    for trials in range(N_ts):
        # For each timestep of the spike train execute a series of Bernoulli trials to generate the spikes:
        trial = np.random.random(data2D.shape)
        data.append((data2D * max_is_present_for / N_ts > trial).astype(dtype=np.uint8))
    res = np.array(data, dtype=np.uint8)  # (Ns, examples, data)
    res = res.swapaxes(0, 1)  # (examples, Ns, data)
    return res


def load(name, limit=2):
    try:
        with open(name + '.pkl', 'rb') as f:
            object = pickle.load(f)
        return object
    except:
        if limit == 0:
            return None
        else:
            found = load('../'+name, limit-1)
            if found is None:
                print('Cannot find', name, 'Current working directory:', os.getcwd())
            return found


def find(name, limit=2):
    """Check if path (file or directory) exists somewhere in the directory tree, up to given distance limit, default 2.

    :param name: path checked for existence
    :param limit: number of directory levels to traverse while searching for the path
    :return: valid existing string path if found, otherwise None
    """
    if os.path.exists(name):
        return name
    else:
        if limit == 0:
            return None
        else:
            found = find('../'+name, limit-1)
            if found is None:
                print('Cannot find', name)
            return found
