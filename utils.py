import os
import numpy as np
import h5py
import json
import torch
from torch.utils.data import Dataset
from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

class CaptionDataset(Dataset):
    def __init__(self, data_folder, data_name, transform):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        """
        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']
        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, 'captions_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, 'captions_length_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
        # Total number of datapoints
        self.dataset_size = len(self.captions)
        self.transform = transform

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        return img, caption, caplen
    
    def __len__(self):
        return self.dataset_size

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))