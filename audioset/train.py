import pandas as pd
import numpy as np
from zmq import device
from albumentations import Compose, ShiftScaleRotate, GridDistortion
from albumentations.pytorch import ToTensor
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import argparse
from utils import AudioDataset, Task5Model, configureTorchDevice, getSampleRateString
from augmentation.SpecTransforms import TimeMask, FrequencyMask, RandomCycle

from config import feature_type, num_frames, seed, permutation, batch_size, num_workers, num_classes, learning_rate, amsgrad, patience, verbose, epochs, workspace, sample_rate


def run(workspace, feature_type, num_frames, perm, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs('{}/model'.format(workspace), exist_ok=True)

    folds = []
    for i in range(5):
        folds.append(pd.read_csv(
            '{}/split/fold_{}_c.txt'.format(workspace, i), delimiter=" ", header=None))

    train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])
    valid_df = folds[perm[3]]
    # test_df = folds[perm[4]]

    spec_transforms = transforms.Compose([
        TimeMask(),
        FrequencyMask(),
        RandomCycle()
    ])

    albumentations_transform = Compose([
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.5),
        GridDistortion(),
        ToTensor()
    ])

    # Create the datasets and the dataloaders

    train_dataset = AudioDataset(workspace, train_df, feature_type=feature_type,
                                 perm=perm,
                                 resize=num_frames,
                                 image_transform=albumentations_transform,
                                 spec_transform=spec_transforms)

    valid_dataset = AudioDataset(
        workspace, valid_df, feature_type=feature_type, perm=perm, resize=num_frames)

    val_loader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    # Define the device to be used
    device = configureTorchDevice()
    # Instantiate the model
    model = Task5Model(num_classes).to(device)

    # Define optimizer, scheduler and loss criteria
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, verbose=verbose)
    criterion = nn.CrossEntropyLoss()

    train_loss_hist = []
    valid_loss_hist = []
    lowest_val_loss = np.inf
    epochs_without_new_lowest = 0

    for i in range(epochs):
        print('Epoch: ', i)

        this_epoch_train_loss = 0
        for sample in tqdm(train_loader):

            inputs = sample['data'].to(device)
            label = sample['labels'].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                model = model.train()
                outputs = model(inputs)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                this_epoch_train_loss += loss.detach().cpu().numpy()

        this_epoch_valid_loss = 0
        for sample in tqdm(val_loader):
            inputs = sample['data'].to(device)
            labels = sample['labels'].to(device)
            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                this_epoch_valid_loss += loss.detach().cpu().numpy()

        this_epoch_train_loss /= len(train_df)
        this_epoch_valid_loss /= len(valid_df)

        train_loss_hist.append(this_epoch_train_loss)
        valid_loss_hist.append(this_epoch_valid_loss)

        if this_epoch_valid_loss < lowest_val_loss:
            lowest_val_loss = this_epoch_valid_loss
            torch.save(model.state_dict(), '{}/model/{}/model_{}_{}'.format(workspace, getSampleRateString(sample_rate),
                       feature_type, str(perm[0])+str(perm[1])+str(perm[2])))
            epochs_without_new_lowest = 0
        else:
            epochs_without_new_lowest += 1

        if epochs_without_new_lowest >= 25:
            break

        print(this_epoch_train_loss, this_epoch_valid_loss)

        scheduler.step(this_epoch_valid_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-w', '--workspace', type=str, default=workspace)
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-n', '--num_frames', type=int, default=num_frames)
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', default=permutation)
    parser.add_argument('-s', '--seed', type=int, default=seed)
    args = parser.parse_args()
    run(args.workspace, args.feature_type,
        args.num_frames, args.permutation, args.seed)
