import numpy as np
from tqdm import tqdm
import pickle
import os
import argparse
from config import feature_type, num_bins, sample_rate
from utils import dataSampleRateString, getSampleRateString
def run(feature_type, num_bins):

    # Load and prepare data
    with open('./metadata/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    train_df = metadata['coarse_train']
    file_list = list(set(train_df.index.tolist()))

    mean = np.zeros((num_bins,))
    M2 = np.zeros((num_bins,))

    data_folder = dataSampleRateString('train', sample_rate)

    n = 0
    for file_name in tqdm(file_list):
        data = np.load('./data/{}/{}/{}.npy'.format(feature_type, data_folder, file_name))
        x = data.mean(axis=1)
        n += 1
        delta = x - mean
        mean = mean + delta/n
        M2 = M2 + delta*(x - mean)

    variance = M2/(n - 1)
    stdev = np.sqrt(variance)

    folder = f'./data/statistics/{getSampleRateString(sample_rate)}'
    os.makedirs(folder, exist_ok = True)
    np.save(f'{folder}/channel_means_{feature_type}', mean)
    np.save(f'{folder}/channel_stds_{feature_type}', stdev)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-n', '--num_bins', type=int, default=num_bins)
    args = parser.parse_args()
    run(args.feature_type, args.num_bins)

