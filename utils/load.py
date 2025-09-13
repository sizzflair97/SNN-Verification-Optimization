from . import CFG
from .dictionary_mnist import TImageBatch, TLabelBatch
from mnist import MNIST
from torchvision.datasets import FashionMNIST
import numpy as np
import typing

def load_mnist(cfg:CFG) -> tuple[TImageBatch, TLabelBatch, TImageBatch, TLabelBatch]:
    # Parameter setting
    num_steps = cfg.num_steps
    GrayLevels = 255  # Image GrayLevels
    cats = [*range(10)]

    # General variables
    images = []  # To keep training images
    labels = []  # To keep training labels
    images_test = []  # To keep test images
    labels_test = []  # To keep test labels

    # loading MNIST dataset
    mndata = MNIST("data/mnist/MNIST/raw/")

    Images, Labels = mndata.load_training()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images.append(
                np.floor(
                    (GrayLevels - Images[i].reshape(28, 28))
                    * (num_steps - 1)
                    / GrayLevels
                ).astype(int)
            )
            labels.append(cats.index(Labels[i]))
    Images, Labels = mndata.load_testing()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images_test.append(
                np.floor(
                    (GrayLevels - Images[i].reshape(28, 28))
                    * (num_steps - 1)
                    / GrayLevels
                ).astype(int)
            )
            labels_test.append(cats.index(Labels[i]))

    del Images, Labels

    # images contain values within [0,num_steps]
    images = typing.cast(TImageBatch, np.asarray(images))
    labels = typing.cast(TLabelBatch, np.asarray(labels))
    images_test = typing.cast(TImageBatch, np.asarray(images_test))
    labels_test = typing.cast(TLabelBatch, np.asarray(labels_test))

    return images, labels, images_test, labels_test

def load_fmnist(cfg:CFG) -> tuple[TImageBatch,TLabelBatch,TImageBatch,TLabelBatch]:
    # Parameter setting
    num_steps = cfg.num_steps
    GrayLevels = 255  # Image GrayLevels
    cats = [*range(10)]

    # General variables
    images = []  # To keep training images
    labels = []  # To keep training labels
    images_test = []  # To keep test images
    labels_test = []  # To keep test labels

    # loading FMNIST dataset
    fmdata = FashionMNIST('./data/', train=True, download=True)

    Images, Labels = fmdata.data.numpy(), fmdata.targets.numpy()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            images.append(np.floor((GrayLevels - Images[i].reshape(28, 28).astype(int)) * (num_steps-1) / GrayLevels).astype(int))
            labels.append(cats.index(Labels[i]))
    Images, Labels = fmdata.data.numpy(), fmdata.targets.numpy()
    Images = np.array(Images)
    for i in range(len(Labels)):
        if Labels[i] in cats:
            # images_test.append(TTT[i].reshape(28,28).astype(int))
            images_test.append(np.floor((GrayLevels - Images[i].reshape(28, 28).astype(int)) * (num_steps-1) / GrayLevels).astype(int))
            labels_test.append(cats.index(Labels[i]))

    del Images, Labels

    #images contain values within [0,num_steps]
    images = typing.cast(TImageBatch, np.asarray(images))
    labels = typing.cast(TLabelBatch, np.asarray(labels))
    images_test = typing.cast(TImageBatch, np.asarray(images_test))
    labels_test = typing.cast(TLabelBatch, np.asarray(labels_test))
    
    return images, labels, images_test, labels_test