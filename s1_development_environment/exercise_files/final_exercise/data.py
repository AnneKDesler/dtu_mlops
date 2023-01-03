import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    path = "C:/Users/anned/OneDrive - Danmarks Tekniske Universitet/Uni/MLOps/dtu_mlops/data/corruptmnist"

    data_train = np.load(path + "/train_0.npz")
    train = [[torch.from_numpy(data_train["images"]).float(),torch.from_numpy(data_train["labels"])]]
    data_train = np.load(path + "/train_1.npz")
    train.append([torch.from_numpy(data_train["images"]).float(),torch.from_numpy(data_train["labels"])])
    data_train = np.load(path + "/train_2.npz")
    train.append([torch.from_numpy(data_train["images"]).float(),torch.from_numpy(data_train["labels"])])
    data_train = np.load(path + "/train_3.npz")
    train.append([torch.from_numpy(data_train["images"]).float(),torch.from_numpy(data_train["labels"])])
    data_train = np.load(path + "/train_4.npz")
    train.append([torch.from_numpy(data_train["images"]).float(),torch.from_numpy(data_train["labels"])])
    data_test = np.load(path + "/test.npz")
    test = [[torch.from_numpy(data_test["images"]).float(),torch.from_numpy(data_test["labels"])]]
    return train, test

train, test = mnist()