import numpy as np


def load_clean_dataset():
    return _load_dataset(
        './dataset/intro2ML-coursework1/wifi_db/clean_dataset.txt')


def load_noisy_dataset():
    return _load_dataset(
        './dataset/intro2ML-coursework1/wifi_db/noisy_dataset.txt')


def _load_dataset(filepath):
    dataset = np.loadtxt(filepath, dtype=np.float16)
    return dataset
