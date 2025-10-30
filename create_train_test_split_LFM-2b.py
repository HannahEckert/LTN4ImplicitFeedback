from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch

from helpers.metrics import create_popularity_bins, join_interaction_with_country

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train test splits creation')
    
    # Data paths
    parser.add_argument('--experiments_folder', type=str, default='data_LFM-2b/data',
                       help='Path to the experiments folder containing dataset files')
    parser.add_argument('--dataset_file', type=str, default='dataset.inter',
                       help='Name of the interactions dataset file')
    parser.add_argument('--demographics_file', type=str, default='demographics.tsv',
                       help='Name of the demographics file')
    parser.add_argument('--tracks_file', type=str, default='tracks.tsv',
                       help='Name of the tracks file')
    parser.add_argument("--save_splits_path", type=str, default="data_splits")

    return parser.parse_args()
    
def main():
    args = parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    args = parse_args()

    EXPERIMENTS_FOLDER = Path(args.experiments_folder)
    dataset_path = EXPERIMENTS_FOLDER / args.dataset_file
    demographics_path = EXPERIMENTS_FOLDER / args.demographics_file
    tracks_path = EXPERIMENTS_FOLDER / args.tracks_file

    interactions = pd.read_csv(dataset_path, delimiter='\t', header=None, skiprows=1, names=['user_id', 'item_id'])
    tracks_info = pd.read_csv(tracks_path, delimiter='\t', header=None).reset_index()
    tracks_info.columns = ['item_id', 'artist', 'title', 'country']
    demographics = pd.read_csv(demographics_path, delimiter='\t', header=None,
                              names=['country', 'age', 'gender', 'signup_date'])

    tracks_info['country'] = tracks_info['country'].replace('GB', 'UK')
    demographics['country'] = demographics['country'].replace('GB', 'UK')

    tracks_with_popularity = create_popularity_bins(interactions, tracks_info)
    df = join_interaction_with_country(interactions, demographics, tracks_info, tracks_with_popularity)

    #remove rows with missing values
    df = df.dropna()

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Change dtype to category
    df["item_id"] = df["item_id"].astype('category').cat.codes

    n_interactions = len(df)

    df_numerical = df.copy()
    df_numerical["user_country"] = df_numerical["user_country"].astype('category').cat.codes
    df_numerical["gender"] = df_numerical["gender"].astype('category').cat.codes
    df_numerical["artist"] = df_numerical["artist"].astype('category').cat.codes
    df_numerical["artist_country"] = df_numerical["artist_country"].astype('category').cat.codes

    user_features = {}
    for i in df_numerical["user_id"].unique():
        user_features[i] = df_numerical[df_numerical["user_id"]==i].iloc[0,[2,3,4]].values.astype(np.float32)

    item_features = {}
    for i in df_numerical["item_id"].unique():
        item_features[i] = df_numerical[df_numerical["item_id"]==i].iloc[0,[6,8]].values.astype(np.float32)

    all_user_items = {}
    for i in df_numerical["user_id"].unique():
        all_user_items[i] = set(df_numerical[df_numerical["user_id"]==i]["item_id"].values)

    data = df.iloc[:n_interactions, [0,1]].to_numpy()

    data_train = []
    data_test = []
    data_validation = []
    user_ids = np.unique(data[:, 0])
    for user_id in user_ids:
        user_mask = data[:, 0] == user_id
        user_data = data[user_mask]
        
        if len(user_data) > 2:
            data_train.extend(user_data[:-2])
            data_validation.append(user_data[-2])
            data_test.append(user_data[-1])
        if len(user_data) > 1 and len(user_data) <= 2:
            data_train.extend(user_data[:-1])
            data_test.append(user_data[-1])
        if len(user_data) == 1:
            data_train.extend(user_data)


    data_train = np.array(data_train)
    data_test = np.array(data_test)
    data_validation = np.array(data_validation)
    print(f"Train size: {len(data_train)}, Test size: {len(data_test)}, Validation size: {len(data_validation)}")

    all_user_items_train = {}
    for i in np.unique(data_train[:,0]):
        all_user_items_train[i] = set(data_train[data_train[:,0] == i][:,1])

    all_user_items_test = {}
    for i in np.unique(data_test[:,0]):
        all_user_items_test[i] = set(data_test[data_test[:,0] == i][:,1])
    all_user_items_val = {}
    for i in np.unique(data_validation[:,0]):
        all_user_items_val[i] = set(data_validation[data_validation[:,0] == i][:,1])

    #save dictionaries and data splits
    np.save(args.save_splits_path +  '/data_train.npy', data_train)
    np.save(args.save_splits_path +  '/data_test.npy', data_test)
    np.save(args.save_splits_path +  '/data_validation.npy', data_validation)
    torch.save(user_features, args.save_splits_path +  '/user_features.pth')
    torch.save(item_features, args.save_splits_path +  '/item_features.pth')
    torch.save(all_user_items, args.save_splits_path +  '/all_user_items.pth')
    torch.save(all_user_items_train, args.save_splits_path +  '/all_user_items_train.pth')
    torch.save(all_user_items_test, args.save_splits_path +  '/all_user_items_test.pth')
    torch.save(all_user_items_val, args.save_splits_path +  '/all_user_items_val.pth')

if __name__ == '__main__':
    main()