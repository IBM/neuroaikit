import pickle
import os


def _load_pickle(filename):
    with open(os.path.join(os.path.dirname(__file__), filename + '.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data


def JSB():
    """
    Loads harmonized JSB music prediction dataset.
    See N. Boulanger-Lewandowski, Y. Bengio, and P. Vincent,
    “Modeling Temporal Dependencies in High-dimensional Sequences:
    Application to Polyphonic Music Generation and Transcription,”
    in Proceedings of the 29th International Conference on
    International Conference on Machine Learning, 2012.

    :return: (train, valid, test)-tuple with data
    """
    return _load_pickle('JSB_train'), _load_pickle('JSB_valid'), _load_pickle('JSB_test')
