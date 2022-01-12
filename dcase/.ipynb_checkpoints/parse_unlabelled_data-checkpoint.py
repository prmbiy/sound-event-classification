import glob
import pandas as pd
import argparse
import os
import pickle

def run(feature_type, input_folder):
    file_names = [os.path.basename(x)[:-4] for x in glob.glob(f'./data/{feature_type}/{input_folder}/*.wav.npy')]
    
    df = pd.DataFrame({
        'audio_filename': file_names
    }).set_index('audio_filename').sort_index()
    
    os.makedirs(f'./metadata/{input_folder}/', exist_ok = True)
    
    with open(f'./metadata/{input_folder}/metadata.pkl', 'wb') as f:
        pickle.dump({
            'coarse_predict' :df
        }, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing data without labels for prediction.')
    parser.add_argument('-f', '--feature_type', type=str, default='logmelspec')
    parser.add_argument('-i', '--input_folder', type=str, help="Specifies name of folder containing spectrograms of unlabelled data.")
    args = parser.parse_args()
    run(args.feature_type, args.input_folder)