import pickle
import numpy as np
import pandas as pd
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc

import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose
from albumentations.pytorch import ToTensor
from utils import Task5Model, AudioDataset

import argparse

def run(args):

    print(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature type')
    # parser.add_argument('-f', '--feature_type', type=str, default='logmelspec')
    # parser.add_argument('-n', '--num_frames', type=int, default=635)
    args = parser.parse_args()
    run(args)