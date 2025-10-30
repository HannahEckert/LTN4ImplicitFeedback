import torch
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import argparse

from helpers.metrics import create_popularity_bins, join_interaction_with_country

EXPERIMENTS_FOLDER = Path('data')

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Mainstreaminess Masueres')

    #Data params
    parser.add_argument('--experiments_folder', type=str, default='data_LFM-2b/data',
                       help='Path to the experiments folder containing dataset files')
    parser.add_argument('--dataset_file', type=str, default='dataset.inter',
                       help='Name of the interactions dataset file')
    parser.add_argument('--demographics_file', type=str, default='demographics.tsv',
                       help='Name of the demographics file')
    parser.add_argument('--tracks_file', type=str, default='tracks.tsv',
                       help='Name of the tracks file')
    parser.add_argument("--min_user_per_country", type=int, default=0,
                       help="Minimum number of users per country to consider")
    parser.add_argument('--dataset_subsize', type=int, default= np.inf,
                       help='Size of the dataset to use for testing, set to np.inf to use the full dataset')
    
    parser.add_argument("--results_save_file_path", type=Path, default=Path("mainstreaminess_measures"),
                       help="Path to save the results of the mainstreaminess measures")
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU ID to use (-1 for CPU, 0 for first GPU, etc.)')
    
    return parser.parse_args()

def main():
    args = parse_args()

    #create results save folder if it does not exist
    if not args.results_save_file_path.exists():
        args.results_save_file_path.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {args.results_save_file_path}")
    else:
        print(f"Results will be saved to existing folder: {args.results_save_file_path}")

    # Set device based on GPU argument
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        # Validate GPU ID is available
        if args.gpu >= torch.cuda.device_count():
            print(f"Warning: GPU {args.gpu} not available, using GPU 0 instead")
            args.gpu = 0
        
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.empty_cache()

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

    df["item_id"] = df["item_id"].astype('category').cat.codes

 
    # Filter out countries with less than min_user_per_country users
    bool_country_size = df.groupby('user_country').size() > args.min_user_per_country
    print(f"Countries with more than {args.min_user_per_country} users: {bool_country_size.sum()} out of {len(bool_country_size)}")
    df = df[df['user_country'].isin(bool_country_size[bool_country_size].index)]
    df = df.reset_index(drop=True)

    user_features = {}
    for i in df["user_id"].unique():
        user_features[i] = df[df["user_id"]==i].iloc[0,[2,3,4]].values


    
    #count of interactions per artist
    APC_global = df.groupby('artist').size()

    # count of interactions per artist per country
    APC_country = df.groupby(['artist', 'user_country']).size().unstack(fill_value=0)



    #global distribution based measure

    APC_per_user = df.groupby(["user_id","artist"]).size().unstack(fill_value=0) 
    APC_per_user_normalized = APC_per_user.divide(APC_per_user.sum(axis=1),axis=0)

    #APC_per_user_normalized = APC_per_user.divide(APC_per_user.sum(axis=1),axis=0)
    APC_global_userwise_normalized = APC_global * (APC_per_user_normalized > 0)
    APC_global_userwise_normalized = APC_global_userwise_normalized.divide(APC_global_userwise_normalized.sum(axis=1), axis=0)

    temp1 = np.log(APC_per_user_normalized.divide(APC_global_userwise_normalized, axis=1))*APC_per_user_normalized

    temp2 = APC_global_userwise_normalized* np.log(APC_global_userwise_normalized/ APC_per_user_normalized)

    temp2[temp2 == np.inf] = 0

    M_global_D_APC = 1-np.mean([1-np.exp(-temp1.sum(axis=1)),1-np.exp(-temp2.sum(axis=1))],axis=0)



    #per country distribution based measure

    APC_per_country = df.groupby(["artist","user_country"]).size().unstack(fill_value=0)

    countries_user = np.array([user_features[ind][0] for ind in np.unique(df["user_id"])])

    APC_per_country_ordered_by_users = APC_per_country[countries_user].T

    APC_per_country_userwise_normalized = APC_per_country_ordered_by_users.to_numpy() * (APC_per_user_normalized > 0)
    APC_per_country_userwise_normalized = APC_per_country_userwise_normalized.divide(APC_per_country_userwise_normalized.sum(axis=1), axis=0)

    temp1 = np.log(np.multiply(np.array(APC_per_user_normalized),np.array(1/APC_per_country_userwise_normalized)))*APC_per_user_normalized
    temp2 = np.log(np.multiply(np.array(APC_per_country_userwise_normalized),np.array(1/APC_per_user_normalized)))*APC_per_country_userwise_normalized

    M_local_D_APC = 1-np.mean([1-np.exp(-temp1.sum(axis=1)),1-np.exp(-temp2.sum(axis=1))],axis=0)

    print("M_local_D_APC mean: ", np.mean(M_local_D_APC))
    print("M_local_D_APC std: ", np.std(M_local_D_APC))

    print("M_global_D_APC mean: ", np.mean(M_global_D_APC))
    print("M_global_D_APC std: ", np.std(M_global_D_APC))

    #save results to csv
    results_user_df = pd.DataFrame({
        'user_id': np.unique(df["user_id"]),
        'M_global_D_APC': M_global_D_APC,
        'M_local_D_APC': M_local_D_APC
    })

    results_artist_df = pd.DataFrame({
        'artist': APC_global.index,
        'APC_global': APC_global.values
    })

    #join with APC_local
    results_artist_df = results_artist_df.join(APC_country, on='artist', how='left')


    results_user_df.to_csv(args.results_save_file_path / 'distribution_based_user_M.csv', index=False)
    results_artist_df.to_csv(args.results_save_file_path / 'distribution_based_artist_M.csv', index=False)
    print("distribution based results saved to:", args.results_save_file_path)




    

        



if __name__ == '__main__':
    main()
