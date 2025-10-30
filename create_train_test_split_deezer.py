import pandas as pd
import numpy as np
import torch
import argparse

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Create train test splits')
    parser.add_argument('--data_path', type=str, default="data_deezer/DEEZER_GLOBAL.inter",
                        help='Path to the Deezer dataset file')
    parser.add_argument("--save_path", type=str, default="data_deezer/data_splits")

    return parser.parse_args()

def main():
    args = parse_args()
    data = pd.read_csv(args.data_path)
    save_path = args.save_path

    #creating empty item features as placeholder (as we only need item features for the case study)
    user_features = {}
    for i in data["user_id:token"].unique():
        user_features[i] = np.array([0,0,0])

    item_features = {}
    for i in data["item_id:token"].unique():
        item_features[i] = [0,0]

    data = data.iloc[:, [0,1]].to_numpy()

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

    np.save(save_path +  '/data_train.npy', data_train)
    np.save(save_path +  '/data_test.npy', data_test)
    np.save(save_path +  '/data_validation.npy', data_validation)

    all_user_items_train = {}
    for i in np.unique(data_train[:,0]):
        all_user_items_train[i] = set(data_train[data_train[:,0] == i][:,1])

    all_user_items_test = {}
    for i in np.unique(data_test[:,0]):
        all_user_items_test[i] = set(data_test[data_test[:,0] == i][:,1])
    all_user_items_val = {}
    for i in np.unique(data_validation[:,0]):
        all_user_items_val[i] = set(data_validation[data_validation[:,0] == i][:,1])

    all_user_items = {}
    for i in np.unique(data[:,0]):
        all_user_items[i] = set(data[data[:,0]==i][:,1])

    torch.save(all_user_items, save_path +  '/all_user_items.pth')
    torch.save(all_user_items_train, save_path +  '/all_user_items_train.pth')
    torch.save(all_user_items_test, save_path +  '/all_user_items_test.pth')
    torch.save(all_user_items_val, save_path +  '/all_user_items_val.pth')

    torch.save(user_features, save_path +  '/user_features.pth')
    torch.save(item_features, save_path +  '/item_features.pth')





if __name__ == '__main__':
    main()
