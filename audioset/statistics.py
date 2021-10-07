import numpy as np 
from tqdm import tqdm
import pickle 
from glob import glob
import argparse
import pandas as pd
import os

def run(workspace, feature_type, num_bins, perm):
#    if '8k' in feature_type:
#        actual_files = glob('{}/{}/*.wav.npy'.format(workspace, feature_type))
#        for file in actual_files:
#            new_filename = file.split('/')[-1].split('.')[0]
#            os.rename(file, '{}/{}/{}.wav.npy'.format(workspace, feature_type, new_filename))
    
    # Load and prepare data
    folds = []
    for i in range(5):
        folds.append(pd.read_csv('{}/split/fold_{}_c.txt'.format(workspace, i), delimiter=" ", header=None))
    
    train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])    
    file_list = train_df[0].unique()
    
    mean = np.zeros((num_bins,))
    M2 = np.zeros((num_bins,))
    
    no_file_count = 0
    n = 0
    for file_name in tqdm(file_list):
        try:
            data = np.load('{}/{}/{}.wav.npy'.format(workspace, feature_type, file_name))
        except FileNotFoundError:
            no_file_count += 1
            print(file_name)
        x = data.mean(axis=1)
        n += 1
        delta = x - mean
        mean = mean + delta/n
        M2 = M2 + delta*(x - mean)

    variance = M2/(n - 1)
    stdev = np.sqrt(variance)
    print(no_file_count)

    os.makedirs('{}/statistics'.format(workspace), exist_ok=True)
    np.save('{}/statistics/channel_means_{}_{}'.format(workspace, feature_type, str(perm[0])+str(perm[1])+str(perm[2])), mean)
    np.save('{}/statistics/channel_stds_{}_{}'.format(workspace, feature_type, str(perm[0])+str(perm[1])+str(perm[2])), stdev)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-w', '--workspace', type=str)
    # parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-f', '--feature_type', type=str, default='logmelspec')
    parser.add_argument('-n', '--num_bins', type=int, default=128)
    parser.add_argument('-p', '--permutation', type=int, nargs='+', required=True)
    args = parser.parse_args()
    run(args.workspace, args.feature_type, args.num_bins, args.permutation)

