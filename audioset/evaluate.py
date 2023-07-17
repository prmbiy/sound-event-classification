import enum
import matplotlib.pyplot as plt
from matplotlib import use
import pandas as pd
import os
import sklearn
import numpy as np
import pandas as pd
from SKCRNN import SKNet50, CRNN9
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, classification_report, f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import argparse
from utils import AudioDataset, Task5Model, configureTorchDevice, getSampleRateString
from config import target_names, feature_type, num_frames, permutation, batch_size, num_workers, num_classes, sample_rate, workspace, use_cbam, seed, use_resampled_data
from glob import glob

__author__ = "Andrew Koh Jin Jie, Anushka Jain and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Param Biyani"
__email__ = "parambiyani8@gmail.com"
__status__ = "Development"

class_mapping = {}
for i, target in enumerate(target_names):
    class_mapping[target] = i


def run(workspace, feature_type, num_frames, one_speech, perm, model_arch, use_cbam, expt_name, cf):

    if use_resampled_data:
        train_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/train/*.wav.npy'.format(workspace,
                                                                                                             feature_type, getSampleRateString(sample_rate))))]
        
        train_list_length = len(train_list)
        for i in range(0, train_list_length):
            if train_list[train_list_length - i - 1].startswith("Silent") or "others" in train_list[train_list_length - i - 1]:
                del train_list[train_list_length - i - 1]
                
        val_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/valid/*.wav.npy'.format(workspace,
                               feature_type, getSampleRateString(sample_rate))))]
        
        val_list_length = len(val_list)
        for i in range(0, val_list_length):
            if val_list[val_list_length - i - 1].startswith("Silent") or "others" in val_list[val_list_length - i - 1]:
                del val_list[val_list_length - i - 1]
                
        test_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/test/*.wav.npy'.format(workspace,
                               feature_type, getSampleRateString(sample_rate))))]
        
        test_list_length = len(test_list)
        for i in range(0, test_list_length):
            if test_list[test_list_length - i - 1].startswith("Silent") or "others" in test_list[test_list_length - i - 1]:
                del test_list[test_list_length - i - 1]
        
        train_df = pd.DataFrame(train_list)
        valid_df = pd.DataFrame(val_list)
        test_df = pd.DataFrame(test_list)
    else:
        folds = []
        for i in range(5):
            folds.append(pd.read_csv(
                '{}/split/fold_{}_c.txt'.format(workspace, i), delimiter=" ", header=None))

        train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])
        valid_df = folds[perm[3]]
        test_df = folds[perm[4]]

    # Create the datasets and the dataloaders
    train_dataset = AudioDataset(
        workspace, train_df,"train", one_speech=one_speech, feature_type=feature_type, perm=perm, resize=num_frames)
    valid_dataset = AudioDataset(
        workspace, valid_df,"valid", one_speech=one_speech, feature_type=feature_type, perm=perm, resize=num_frames)
    test_dataset = AudioDataset(
        workspace, test_df,"test", one_speech=one_speech, feature_type=feature_type, perm=perm, resize=num_frames)
    test_loader = DataLoader(test_dataset, batch_size,
                             shuffle=False, num_workers=num_workers)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    device = configureTorchDevice()

    # Instantiate the model
    model = Task5Model(num_classes, model_arch, use_cbam=use_cbam).to(device)
    # model = SKNet50().to(device)
    # model = CRNN9().to(device)
    model_path = '{}/model/{}/{}/model_{}_{}_{}'.format(workspace, expt_name, getSampleRateString(
        sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2]), model_arch)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    print(f'Using {model_arch} model from {model_path}.')

    y_pred = []
    for sample in test_loader:
        inputs = sample['data'].to(device)
        labels = sample['labels'].to(device)

        with torch.set_grad_enabled(False):
            model = model.eval()
            outputs = model(inputs)
            for i in range(len(outputs)):
                curr = outputs[i]
                arg = torch.argmax(curr)
                y_pred.append(arg.detach().cpu().numpy())
    y_true = []
    y_true_class = []
    y_pred_class = []

    for i in y_pred:
            y_pred_class.append(list(class_mapping.keys())[list(class_mapping.values()).index(i)])
    for index, row in test_df.iterrows():
        # if row[0].split('-')[0].__contains__("_"):
        #     class_name = row[0].split('-')[0].split("_")[0]
        # else:
        #     class_name = row[0].split('-')[0]
        # if one_speech:
        #     if class_name == 'Female speech':
        #         class_name = 'Male speech'
        class_name = row[0].split('-')[0]
        
        y_true.append(class_mapping[class_name])
        y_true_class.append(class_name)

    print(f'Including other class:')
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Micro F1 Score: {f1_score(y_true, y_pred, average='micro')}")
    print(f"Macro F1 Score: {f1_score(y_true, y_pred, average='macro')}")
    print(f'Accuracy Score: {accuracy_score(y_true, y_pred)}')
    print(y_true[:5], y_pred[:5])

    labels_copy = list(class_mapping.keys())
    cm = confusion_matrix(y_true_class, y_pred_class, labels=labels_copy)
    with np.errstate(divide='ignore'):   
        cm = np.log2(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_copy)
    disp.plot()
    title = 'Confusion matrix for ' + expt_name
    plt.title(title)
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.savefig("figure")
    
    return
    y_true_new = []
    y_pred_new = []

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != 10:
            y_true_new.append(yt)
            y_pred_new.append(yp)

    print(len(y_true), len(y_true_new), np.unique(
        y_true_new, return_counts=True))
    print(f'Excluding other class:')
    print(classification_report(y_true_new, y_pred_new, digits=4))
    print(
        f"Micro F1 Score: {f1_score(y_true_new, y_pred_new, average='micro')}")
    print(
        f"Macro F1 Score: {f1_score(y_true_new, y_pred_new, average='macro')}")
    print(f'Accuracy Score: {accuracy_score(y_true_new, y_pred_new)}')
    print(y_true_new[:5], y_pred_new[:5])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-en', '--expt_name', type=str, required=True)
    parser.add_argument('-w', '--workspace', type=str, default=workspace)
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-n', '--num_frames', type=int, default=num_frames)
    parser.add_argument('-ma', '--model_arch', type=str, default='mobilenetv2')
    parser.add_argument('-os', '--one_speech', type=bool, default=False)
    parser.add_argument('-cbam', '--use_cbam', action='store_true')
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', default=permutation)
    args = parser.parse_args()
    run(args.workspace, args.feature_type, args.num_frames, args.one_speech,
        args.permutation, args.model_arch, args.use_cbam, args.expt_name, args.cf)
