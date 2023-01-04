import torch
import numpy as np
from torchvision import transforms

def mnist():
    # exchange with the corrupted mnist dataset
    path = "C:/Users/anned/OneDrive - Danmarks Tekniske Universitet/Uni/MLOps/dtu_mlops/data"

    train = []
    for i in range(5):
        data_train = np.load(path + "/corruptmnist/train_{}.npz".format(i))
        images = data_train["images"]
        mean = images.mean((1,2))[:, np.newaxis,np.newaxis]
        std = images.std((1,2))[:, np.newaxis,np.newaxis]
        images = (images - mean)/std
        images = images[:, np.newaxis,:,:]
        images = torch.from_numpy(images).float()
        labels = torch.from_numpy(data_train["labels"])
        train.append([images,labels])
    data_test = np.load(path + "/corruptmnist/test.npz")
    images = data_test["images"]
    mean = images.mean((1,2))[:, np.newaxis,np.newaxis]
    std = images.std((1,2))[:, np.newaxis,np.newaxis]
    images = (images - mean)/std
    images = images[:, np.newaxis,:,:]
    images = torch.from_numpy(images).float()
    labels = torch.from_numpy(data_test["labels"])
    test = [[images,labels]]
    return train, test

train, test = mnist()