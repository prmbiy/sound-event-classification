import pickle
import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import DataLoader
from utils import Task5Model, AudioDataset, enableGpuIfExists
from config import cuda, batch_size, num_classes
import argparse

def run(feature_type, num_frames, input_folder):
    with open(f'./metadata/{input_folder}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    predict_df = metadata["coarse_predict"]
    predict_files = []

    predict_dataset = AudioDataset(predict_df, 'logmelspec', resize=num_frames, data_type='predict', input_folder=input_folder)
    predict_loader = DataLoader(predict_dataset, batch_size, shuffle=False)

    device = enableGpuIfExists(cuda)

    model = Task5Model(num_classes).to(device)
    model.load_state_dict(torch.load('./models/model_{}'.format(feature_type)))

    preds = []
    for sample in predict_loader:
            inputs = sample['data'].to(device)
            file_name = sample['file_name']

            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                for j in range(outputs.shape[0]):
                    preds.append(outputs[j,:].detach().cpu().numpy())
                    predict_files.append(file_name[j])

    preds = np.array(preds)
    output_df = pd.DataFrame(
        preds, columns=[
            '1_engine', '2_machinery-impact', '3_non-machinery-impact',
            '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog'])

    output_df['audio_filename'] = pd.Series(
        predict_files,
        index=output_df.index)
    
    
    output_df = output_df.sort_values('audio_filename', ignore_index=True)

    #output_df.to_csv('./models/pred_eval_testing_3s.csv', index=False)
    os.makedirs(f"./models/{input_folder}/", exist_ok = True)
    output_df.to_csv(f"./models/{input_folder}/pred.csv", index=False)
    print(f"Predictions saved in './models/{input_folder}/pred.csv'.")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='For making predictions')
    parser.add_argument('-f', '--feature_type', type=str, default='logmelspec')
    parser.add_argument('-n', '--num_frames', type=int, default=635)
    parser.add_argument('-i', '--input_folder', type=str, help="Specifies name of folder containing spectrograms of unlabelled data.")
    args = parser.parse_args()
    run(args.feature_type, args.num_frames, args.input_folder)

#  Predictions seem to match for: "07_003142.wav"
# "04_001924.wav"
# "00_002112.wav"