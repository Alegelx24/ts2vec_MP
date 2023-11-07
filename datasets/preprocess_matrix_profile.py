import pandas as pd
import pickle
import argparse
import glob
import os

def read_and_prepare_data(files_pattern, currentIndex):
    dataset = []
    for file_name in glob.glob(files_pattern):
        df = pd.read_csv(file_name)
        # Here we assume the columns are named 'left_mp' and 'all_timestamp'
        # If the names are different, adjust them accordingly.
        data_entry = {
            'left_MP': df['left_MP'].tolist()[currentIndex:],
            'timestamp': ( df['all_timestamp']).tolist()[currentIndex:],

        }
        dataset.append(data_entry)
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to the folder containing CSV files', default='datasets')
    parser.add_argument('-o', '--output', type=str, help='Output pkl file name', default='mp.pkl')
    args = parser.parse_args()

    # Change the pattern to match your CSV files
    files_pattern = os.path.join(args.path, '*.csv')
    dataset = read_and_prepare_data(files_pattern, currentIndex=5000)

    # Save to a .pkl file
    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)
