import enum
import pandas as pd
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
import argparse
from utils import AudioDataset, Task5Model, configureTorchDevice, getSampleRateString
from config import target_names, feature_type, num_frames, permutation, batch_size, num_workers, num_classes, sample_rate, workspace

class_mapping = {}
for i, target in enumerate(target_names):
    class_mapping[target] = i


def run(workspace, feature_type, num_frames, perm):

    folds = []
    for i in range(5):
        folds.append(pd.read_csv(
            '{}/split/fold_{}_c.txt'.format(workspace, i), delimiter=" ", header=None))

    train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])
    valid_df = folds[perm[3]]
    test_df = folds[perm[4]]

    # Create the datasets and the dataloaders
    train_dataset = AudioDataset(
        workspace, train_df, feature_type=feature_type, perm=perm, resize=num_frames)
    valid_dataset = AudioDataset(
        workspace, valid_df, feature_type=feature_type, perm=perm, resize=num_frames)
    test_dataset = AudioDataset(
        workspace, test_df, feature_type=feature_type, perm=perm, resize=num_frames)
    test_loader = DataLoader(test_dataset, batch_size,
                             shuffle=False, num_workers=num_workers)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    device = configureTorchDevice()

    # Instantiate the model
    model = Task5Model(num_classes).to(device)
    model.load_state_dict(torch.load('{}/model/{}/model_{}_{}'.format(workspace, getSampleRateString(sample_rate),
                          feature_type, str(perm[0])+str(perm[1])+str(perm[2])))['model_state_dict'])

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

    for index, row in test_df.iterrows():
        class_name = row[0].split('-')[0]
        y_true.append(class_mapping[class_name])

    print(f'Including other class:')
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Micro F1 Score: {f1_score(y_true, y_pred, average='micro')}")
    print(f"Macro F1 Score: {f1_score(y_true, y_pred, average='macro')}")
    print(f'Accuracy Score: {accuracy_score(y_true, y_pred)}')
    print(y_true[:5], y_pred[:5])

    y_true_new = []
    y_pred_new = []

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != 9 and yp!=9:
            y_true_new.append(yt)
            y_pred_new.append(yp)

    print(len(y_true), len(y_true_new), np.unique(y_true_new, return_counts=True))
    print(f'Excluding other class:')
    print(classification_report(y_true_new, y_pred_new, digits=4))
    print(f"Micro F1 Score: {f1_score(y_true_new, y_pred_new, average='micro')}")
    print(f"Macro F1 Score: {f1_score(y_true_new, y_pred_new, average='macro')}")
    print(f'Accuracy Score: {accuracy_score(y_true_new, y_pred_new)}')
    print(y_true_new[:5], y_pred_new[:5])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-w', '--workspace', type=str, default=workspace)
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-n', '--num_frames', type=int, default=num_frames)
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', default=permutation)
    args = parser.parse_args()
    run(args.workspace, args.feature_type, args.num_frames, args.permutation)
