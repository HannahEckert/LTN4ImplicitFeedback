from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import argparse

from helpers.metrics import create_popularity_bins, join_interaction_with_country

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Countries')

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
    parser.add_argument('--dataset_subsize', type=int, default=np.inf,
                       help='Size of the dataset to use for testing, set to np.inf to use the full dataset')
    parser.add_argument("--properties_to_calculate", nargs='+',
                       help="List of properties to calculate; possible parameters: " \
                       "user_country, item_artist_mapping, countries_percentages, item_country_mapping, " \
                       "number_interactions, country_distributions, artist_country_indicator_matrix")

    #Mainstreaminess params
    parser.add_argument("--results_save_file_path", type=Path, default=Path("country_data"),
                       help="Path to save the country data")

    return parser.parse_args()

def main():
    args = parse_args()

    #create results save folder if it does not exist
    if not args.results_save_file_path.exists():
        args.results_save_file_path.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {args.results_save_file_path}")
    else:
        print(f"Results will be saved to existing folder: {args.results_save_file_path}")


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

    #samler dataset for testing
    if args.dataset_subsize < np.inf:
        df = df[:args.dataset_subsize]
        print(f"Using a smaller dataset with size: {args.dataset_subsize}")

    df["item_id"] = df["item_id"].astype('category').cat.codes


    # Filter out countries with less than min_user_per_country users
    bool_country_size = df.groupby('user_country').size() > args.min_user_per_country
    print(f"Countries with more than {args.min_user_per_country} users: {bool_country_size.sum()} out of {len(bool_country_size)}")
    df = df[df['user_country'].isin(bool_country_size[bool_country_size].index)]
    df = df.reset_index(drop=True)

    if "country_distributions" in args.properties_to_calculate:
        user_country_matrix = pd.crosstab(df['user_id'], df['artist_country'])

        user_country_matrix.to_csv(args.results_save_file_path / "user_country_matrix.csv", index=False)

    if "artist_country_indicator_matrix" in args.properties_to_calculate:
        artist_country_matrix = (pd.crosstab(df['artist'], df['artist_country']) > 0).astype(int)
        APC_global = artist_country_matrix["US"]
        artist_country_matrix["APC_global"] = APC_global
        #rearange columns
        cols = artist_country_matrix.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        artist_country_matrix = artist_country_matrix[cols]
        artist_country_matrix.to_csv(args.results_save_file_path / "artist_country_indicator_matrix.csv", index=True)

    if "user_country" in args.properties_to_calculate:
        user_country_mapping = df[['user_id', 'user_country']].drop_duplicates(subset='user_id')

        #save result
        user_country_mapping.to_csv(args.results_save_file_path / "user_country_mapping.csv", index=False)

    if "item_artist_mapping" in args.properties_to_calculate:
        item_artist_mapping = df[['item_id', 'artist']].drop_duplicates(subset='item_id')

        #save result
        item_artist_mapping.to_csv(args.results_save_file_path / "item_artist_mapping.csv", index=False)

    if "countries_percentages" in args.properties_to_calculate:
        # Get total counts per user
        total_counts = df.groupby("user_id")["artist_country"].count()

        # Get US counts per user
        us_counts = df[df["artist_country"] == "US"].groupby("user_id")["artist_country"].count()

        # Calculate percentage
        percentages_US = (us_counts / total_counts * 100).fillna(0).round(2)

        # Get counts of matches per user
        same_country = df[df["artist_country"] == df["user_country"]].groupby("user_id").size()
        total_counts = df.groupby("user_id").size()

        # Calculate percentage
        percentages_same_country = (same_country / total_counts * 100).fillna(0).round(2)

        percentages_US.to_csv(args.results_save_file_path / "percentages_US.csv", index=False)
        percentages_same_country.to_csv(args.results_save_file_path / "percentages_same_country.csv", index=False)

    if "item_country_mapping" in args.properties_to_calculate:
        item_country_mapping = df[['item_id', 'artist_country']].drop_duplicates(subset='item_id')

        #save result
        item_country_mapping.to_csv(args.results_save_file_path / "item_country_mapping.csv", index=False)

    if "number_interactions" in args.properties_to_calculate:
        number_interactions = df.groupby("user_id")["item_id"].count()
        number_interactions.to_csv(args.results_save_file_path / "number_interactions.csv", index=False)



if __name__ == '__main__':
    main()
