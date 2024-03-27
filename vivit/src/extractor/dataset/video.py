from __future__ import annotations

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import VivitImageProcessor, VivitForVideoClassification

def emotion2int(emotion: str):
    emotion2int = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "fear": 4, "disgust": 5, "surprise": 6}
    return emotion2int[emotion]


class VideoFeaturesDataset(Dataset):
    def __init__(self, features_dict, *, use_cuda=False):
        self.features_dict = features_dict
        self.use_cuda = use_cuda
        self.items = list(features_dict.items())

    def __getitem__(self, index):
        (label, k), features = self.items[index]
        # features = torch.tensor(features[:20], dtype=torch.float32)
        # features = torch.cat([torch.zeros(20 - features.shape[0], features.shape[1]), features])
        if self.use_cuda:
            #print(features["pixel_values"].shape)
            #print (label)
            return features["pixel_values"].cuda(), torch.tensor(label, dtype=torch.int64).cuda()
            # print(features["pixel_values"].shape)
        else:
            return features, torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.items)
    


