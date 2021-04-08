from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os
import pickle

from utils import nested_tensor_from_tensor_list, read_json, under_max

MAX_DIM = 299

class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)

train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class MoleculeCaption(Dataset):
    def __init__(self, data_directory, max_length, limit=1000000, transform=val_transform):
        super().__init__()
        self.data_directory = data_directory
        self.transform = transform
        self.data = pickle.load(open(os.path.join(data_directory,'dataset_img2smi.pkl'), 'rb'))['images'][:limit]
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        caption = np.array(self.data[idx]['sentences'][0]['selfies_ids'])
        caption_length = self.data[idx]['sentences'][0]['selfies_length']
        caption_mask = np.array([False if i < caption_length else True for i in range(len(caption))])
        image = Image.open(os.path.join(self.data[idx]['filepath'], self.data[idx]['filename']))
        image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))
        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, caption_mask
