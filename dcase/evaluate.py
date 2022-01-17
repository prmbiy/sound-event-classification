import pickle
import numpy as np
import pandas as pd
from dcase.utils import getSampleRateString
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc
from config import batch_size, labels, num_classes, sample_rate, feature_type, num_frames

import torch
from torch.utils.data import DataLoader
from utils import Task5Model, AudioDataset, configureTorchDevice,dataSampleRateString

import argparse

with open('./metadata/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

def run(feature_type, num_frames, sample_rate):

    validate_files = []
    valid_df = metadata["coarse_test"]

    valid_dataset = AudioDataset(valid_df, feature_type, resize=num_frames, mode='validate', input_folder=dataSampleRateString('validate', sample_rate))
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)

    device = configureTorchDevice()

    model = Task5Model(num_classes).to(device)
    folder_name = f'./models/{getSampleRateString(sample_rate)}'
    model.load_state_dict(torch.load(f'{folder_name}/model_{feature_type}'))

    preds = []
    for sample in valid_loader:
            inputs = sample['data'].to(device)
            file_name = sample['file_name']

            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                for j in range(outputs.shape[0]):
                    preds.append(outputs[j,:].detach().cpu().numpy())
                    validate_files.append(file_name[j])

    preds = np.array(preds)
    output_df = pd.DataFrame(
        preds, columns=labels)

    output_df['audio_filename'] = pd.Series(
        validate_files,
        index=output_df.index)

    output_df.to_csv(f'{folder_name}/pred.csv', index=False)

    mode = "coarse"
    df_dict = evaluate(f'{folder_name}/pred.csv',
                       './metadata/annotations-dev.csv',
                       './metadata/dcase-ust-taxonomy.yaml',
                       "coarse")

    micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
    macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)
    # Get index of first threshold that is at least 0.5
    thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).nonzero()[0][0]
    print("{} level evaluation:".format(mode.capitalize()))
    print("======================")
    print(" * Micro AUPRC:           {}".format(micro_auprc))
    print(" * Micro F1-score (@0.5): {}".format(eval_df["F"][thresh_0pt5_idx]))
    print(" * Macro AUPRC:           {}".format(macro_auprc))
    print(" * Coarse Tag AUPRC:")

    for coarse_id, auprc in class_auprc.items():
        print("      - {}: {}".format(coarse_id, auprc))

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-n', '--num_frames', type=int, default=num_frames)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-sr', '--sample_rate', type=int,
                        default=sample_rate, choices=[8000, 16000])
    args = parser.parse_args()
    run(args.feature_type, args.num_frames, args.sample_rate)
