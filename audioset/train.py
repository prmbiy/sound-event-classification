#!/usr/bin/env python
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
from utils import AudioDataset, Task5Model, configureTorchDevice, getSampleRateString, BalancedBatchSampler, prism_loss
from SKCRNN import SKNet26, SKNet50, CRNN9
from augmentation.SpecTransforms import TimeMask, FrequencyMask, RandomCycle
from torchsummary import summary
from config import feature_type, num_frames, seed, permutation, batch_size, num_workers, num_classes, learning_rate, amsgrad, patience, verbose, epochs, workspace, sample_rate, early_stopping, grad_acc_steps, model_arch, pann_cnn10_encoder_ckpt_path, pann_cnn14_encoder_ckpt_path, sk_cnn26_encoder_path, resume_training, n_mels, use_cbam, use_resampled_data
# import wandb
import sklearn
from glob import glob

__author__ = "Andrew, Yan Zhen, Anushka and Soham"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

def run(args):
    # wandb.init(project="st-project-sec", entity="sohamtiwari3120")
    # wandb.config.update(args)
    expt_name = args.expt_name
    workspace = args.workspace
    feature_type = args.feature_type
    num_frames = args.num_frames
    perm = args.permutation
    seed = args.seed
    resume_training = args.resume_training
    grad_acc_steps = args.grad_acc_steps
    model_arch = args.model_arch
    use_cbam = args.use_cbam
    use_pna = args.use_pna
    one_speech = args.one_speech
    # print(f'Using cbam: {use_cbam}')
    # print(f'Using pna: {use_pna}')
    if model_arch == 'pann_cnn10':
        pann_cnn10_encoder_ckpt_path = args.pann_cnn10_encoder_ckpt_path
    elif model_arch == 'pann_cnn14':
        pann_cnn14_encoder_ckpt_path = args.pann_cnn14_encoder_ckpt_path
    elif model_arch == 'sk_cnn26':
        sk_cnn26_encoder_path = args.sk_cnn26_encoder_path
    balanced_sampler = args.balanced_sampler

    starting_epoch = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs('{}/model'.format(workspace), exist_ok=True)
    
    if use_resampled_data:
        train_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/train/*.wav.npy'.format(workspace,
                               feature_type, getSampleRateString(sample_rate))))]
        
        train_list_size = len(train_list)
        for i in range(0, train_list_size):
            if train_list[train_list_size - i - 1].startswith("Silent") or "others" in train_list[train_list_size - i - 1]:
                del train_list[train_list_size - i - 1]
                
        val_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/valid/*.wav.npy'.format(workspace,
                               feature_type, getSampleRateString(sample_rate))))]
        
        val_list_size = len(val_list)
        for i in range(0, val_list_size):
            if val_list[val_list_size - i - 1].startswith("Silent") or "others" in val_list[val_list_size - i - 1]:
                del val_list[val_list_size - i - 1]
                
        # train_list, val_list = sklearn.model_selection.train_test_split(file_list, train_size=0.8, random_state = seed)
        train_df = pd.DataFrame(train_list)
        valid_df = pd.DataFrame(val_list)
    else:
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

    train_dataset = AudioDataset(workspace, train_df,"train", one_speech=one_speech, feature_type=feature_type,
                                 perm=perm,
                                 resize=num_frames,
                                 image_transform=albumentations_transform,
                                 spec_transform=spec_transforms)

    valid_dataset = AudioDataset(
        workspace, valid_df,"valid", one_speech=one_speech, feature_type=feature_type, perm=perm, resize=num_frames)

    val_loader = DataLoader(valid_dataset, batch_size,
                            shuffle=False, num_workers=num_workers)
    # print(f'Using balanced_sampler = {balanced_sampler}')
    if balanced_sampler:
        train_loader = DataLoader(train_dataset, batch_size, sampler=BalancedBatchSampler(train_df), num_workers=num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    # Define the device to be used
    device = configureTorchDevice()
    # Instantiate the model
    if model_arch == 'mobilenetv2' or model_arch == 'mobilenetv3':
        model = Task5Model(num_classes, model_arch, use_cbam=use_cbam).to(device)
    elif model_arch == 'pann_cnn10':
        model = Task5Model(num_classes, model_arch, pann_cnn10_encoder_ckpt_path=pann_cnn10_encoder_ckpt_path, use_cbam=use_cbam, use_pna = use_pna).to(device)
    elif model_arch == 'pann_cnn14':
        model = Task5Model(num_classes, model_arch, pann_cnn14_encoder_ckpt_path=pann_cnn14_encoder_ckpt_path, use_cbam=use_cbam, use_pna = use_pna).to(device)
    elif model_arch == 'sk_cnn26':
        model = SKNet26().to(device)
    elif model_arch == 'sk_cnn50':
        model = SKNet50().to(device)
    elif model_arch == 'crnn9':
        model = CRNN9().to(device)
    print(f'Using {model_arch} model.')
#     summary(model, (1, n_mels, num_frames))
    # wandb.watch(model, log_freq=100)
    folderpath = '{}/model/{}/{}'.format(workspace, expt_name,
                                      getSampleRateString(sample_rate))
    os.makedirs(folderpath, exist_ok=True)
    model_path = '{}/model_{}_{}_{}'.format(folderpath,
                                            feature_type, str(perm[0])+str(perm[1])+str(perm[2]), model_arch)

    # Define optimizer, scheduler and loss criteria
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, amsgrad=amsgrad)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, verbose=verbose)
    criterion = nn.CrossEntropyLoss()

    train_loss_hist = []
    valid_loss_hist = []
    lowest_val_loss = np.inf
    epochs_without_new_lowest = 0

    if resume_training and os.path.exists(model_path):
        print(f'resume_training = {resume_training} using path {model_path}')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    optimizer.zero_grad()
    for i in range(starting_epoch, starting_epoch+epochs):
        print('Epoch: ', i)

        this_epoch_train_loss = 0
        batch = 0
        for sample in tqdm(train_loader):
            batch += 1
            inputs = sample['data'].to(device)
            label = sample['labels'].to(device)

            with torch.set_grad_enabled(True):
                model = model.train()
                # print(inputs.shape)
                # print(inputs)
                outputs = model(inputs)
                loss = criterion(outputs, label) + prism_loss(outputs)
                loss.backward()
                if batch % grad_acc_steps == 0 or batch % len(train_loader) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                this_epoch_train_loss += loss.detach().cpu().numpy()
        
        this_epoch_valid_loss = 0
        for sample in tqdm(val_loader):
            inputs = sample['data'].to(device)
            labels = sample['labels'].to(device)
            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(inputs)
                loss = criterion(outputs, labels) + prism_loss(outputs)
                this_epoch_valid_loss += loss.detach().cpu().numpy()

        this_epoch_train_loss /= len(train_df)
        this_epoch_valid_loss /= len(valid_df)
        # wandb.log({"train":{
        #     "loss": this_epoch_train_loss
        # }})
        # wandb.log({"validation":{
        #     "loss": this_epoch_valid_loss
        # }})
        print(f"train_loss = {this_epoch_train_loss}, val_loss={this_epoch_valid_loss}")
        train_loss_hist.append(this_epoch_train_loss)
        valid_loss_hist.append(this_epoch_valid_loss)

        if this_epoch_valid_loss < lowest_val_loss:
            lowest_val_loss = this_epoch_valid_loss
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path)
            print(f'Saving model state at epoch: {i}.')
            epochs_without_new_lowest = 0
        else:
            epochs_without_new_lowest += 1

        if epochs_without_new_lowest >= early_stopping:
            break

        scheduler.step(this_epoch_valid_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-en', '--expt_name', type=str, required=True)
    parser.add_argument('-w', '--workspace', type=str, default=workspace)
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-ma', '--model_arch', type=str, default=model_arch)
    parser.add_argument('-cp10', '--pann_cnn10_encoder_ckpt_path',
                        type=str, default=pann_cnn10_encoder_ckpt_path)
    parser.add_argument('-cp14', '--pann_cnn14_encoder_ckpt_path',
                        type=str, default=pann_cnn14_encoder_ckpt_path)
    parser.add_argument('-sk26', '--sk_cnn26_encoder_path',
                        type=str, default=sk_cnn26_encoder_path)
    parser.add_argument('-n', '--num_frames', type=int, default=num_frames)
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', default=permutation) 
    parser.add_argument('-s', '--seed', type=int, default=seed)
    parser.add_argument('-rt', '--resume_training', action='store_true')
    parser.add_argument('-bs', '--balanced_sampler', type=bool, default=False)
    parser.add_argument('-os', '--one_speech', type=bool, default=False)
    parser.add_argument('-cbam', '--use_cbam', action='store_true')
    parser.add_argument('-pna', '--use_pna', action='store_true')
    parser.add_argument('-ga', '--grad_acc_steps',
                        type=int, default=grad_acc_steps)
    args = parser.parse_args()
    run(args)
