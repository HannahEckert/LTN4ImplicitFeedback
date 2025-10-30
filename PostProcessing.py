import pandas as pd
import numpy as np
import torch
import tqdm
import argparse

torch.manual_seed(44)
np.random.seed(44)
torch.cuda.manual_seed(44) 

def parse_args():
    parser = argparse.ArgumentParser(description='Post Processing')

    parser.add_argument('--path_baseline', default = "final_results_bce/Baseline/", type=str, help='Path to the baseline results')
    parser.add_argument("--k", default=10, type = int , help="Number of recommended items")
    parser.add_argument("--path_mainstreaminess_measures", default="results_full_sample/mainstreaminess_measures/", 
                        type=str, help="Path to the mainstreaminess measures")
    parser.add_argument("--path_country_data", default="results_full_sample/country_data/", 
                        type=str, help="Path to the country data")
    parser.add_argument("--path_output", default="final_results_bce/PostProcessing/", 
                        type=str, help="Path to save the results")
    parser.add_argument("--l", default=0.01, type=float, help="Trade-off parameter for fairness")
    parser.add_argument("--bias_local_weight", default=0.5, type=float, help="Weight for local bias vs US bias")

    return parser.parse_args()

def main():
    args = parse_args()
    path_baseline = args.path_baseline
    k = args.k
    path_mainstreaminess_measures = args.path_mainstreaminess_measures
    path_country_data = args.path_country_data

    M = pd.read_csv(path_mainstreaminess_measures + "distribution_based_user_M.csv")
    item_country_mapping = pd.read_csv(path_country_data + "item_country_mapping.csv")
    user_country_mapping = pd.read_csv(path_country_data + "user_country_mapping.csv")
    true_percentages_US = pd.read_csv(path_country_data + "percentages_US.csv")
    true_percentages_domestic = pd.read_csv(path_country_data + "percentages_same_country.csv")

    df = M.copy()
    df = df.join(user_country_mapping.set_index("user_id"), on="user_id")
    df.index = df["user_id"]
    df["true_US"] = np.array(true_percentages_US)[df["user_id"]]
    df["true_domestic"] = np.array(true_percentages_domestic)[df["user_id"]]
    #rename columns
    df = df.rename(columns={"M_global_D_APC" : "M_global", "M_local_D_APC" : "M_local"})
    # only countries with more than 100 users
    df = df[df["user_country"].isin(df["user_country"].value_counts()[df["user_country"].value_counts()>100].index)]
    # Convert to category and get the mapping
    item_country_cat = item_country_mapping["artist_country"].astype('category')
    country_categories = item_country_cat.cat.categories  # Index of country names

    us_index = np.where(country_categories == "US")[0][0]

    # Ensure the categories are the same as in item_country_cat
    user_country_cat = pd.Categorical(user_country_mapping["user_country"], categories=country_categories)
    user_country_codes = user_country_cat.codes  # -1 means not found in categories

    df["user_country_codes"] = user_country_codes[df["user_id"]]

    scores1 = torch.load(path_baseline + "/1/results_scores.pth", weights_only = False)
    scores2 = torch.load(path_baseline + "/2/results_scores.pth", weights_only = False)
    scores3 = torch.load(path_baseline + "/3/results_scores.pth", weights_only = False)
    random_scores = torch.rand_like(torch.tensor(scores1))/10000000
    scores1 = torch.tensor(scores1) + random_scores
    scores2 = torch.tensor(scores2) + random_scores
    scores3 = torch.tensor(scores3) + random_scores
    scores1 = scores1.numpy()
    scores2 = scores2.numpy()
    scores3 = scores3.numpy()
    candidates1 = torch.load(path_baseline + "/1/results_candidates.pth", weights_only=False)
    candidates2 = torch.load(path_baseline + "/2/results_candidates.pth", weights_only=False)
    candidates3 = torch.load(path_baseline + "/3/results_candidates.pth", weights_only=False)
    scores1 = scores1[df["user_id"]]
    scores2 = scores2[df["user_id"]]
    scores3 = scores3[df["user_id"]]
    candidates1 = candidates1[df["user_id"]]
    candidates2 = candidates2[df["user_id"]]
    candidates3 = candidates3[df["user_id"]]

    scores1 = pd.DataFrame(scores1, index=df["user_id"])
    scores2 = pd.DataFrame(scores2, index=df["user_id"])
    scores3 = pd.DataFrame(scores3, index=df["user_id"])
    countries_candidates1 = pd.DataFrame(candidates1[:,:,2].astype(int), index=df["user_id"])
    countries_candidates2 = pd.DataFrame(candidates2[:,:,2].astype(int), index=df["user_id"])
    countries_candidates3 = pd.DataFrame(candidates3[:,:,2].astype(int), index=df["user_id"])

    user_country_matrix = pd.read_csv(path_country_data +"/user_country_matrix.csv")
    user_country_matrix= user_country_matrix.iloc[df["user_id"]]
    
    def efficient_brute_force_local_and_global(scores, user, countries_candidates, k=10, l=0.5, bias_local_weight=1):

        n_candidates = scores.shape[1]
        user_scores = scores[scores.index == user].values.flatten()
        user_countries = countries_candidates[countries_candidates.index == user].values.flatten()

        # Get the specific user's country code
        user_country_code = df["user_country_codes"][user]
        
        # Precompute bias for each individual item
        true_local_rate = df["true_domestic"][user]/100
        true_us_rate = df["true_US"][user] / 100

        # Track selected indices and remaining candidates
        selected_indices = []
        available_indices = set(range(n_candidates))
        
        # Keep running totals for efficiency
        total_score = 0.0
        total_local_count = 0
        total_us_count = 0
        
        for step in range(k):
            best_value = -np.inf
            best_idx = None
            
            for idx in available_indices:
                # Calculate incremental updates
                new_total_score = total_score + user_scores[idx]
                new_total_local_count = total_local_count + (user_countries[idx] == user_country_code)
                new_total_us_count = total_us_count + (user_countries[idx] == us_index)
                new_set_size = step + 1

                # Calculate new bias and score
                new_mean_score = new_total_score / new_set_size
                new_local_rate = new_total_local_count / new_set_size
                new_bias_local = abs(new_local_rate - true_local_rate)
                new_us_rate = new_total_us_count / new_set_size
                new_bias_us = abs(new_us_rate - true_us_rate)
                new_bias = (bias_local_weight*new_bias_local + (1-bias_local_weight)*new_bias_us) 

                current_value = ((1 - l) * new_mean_score - l * new_bias)

                if current_value > best_value:
                    best_value = current_value
                    best_idx = idx
            
            # Update running totals
            selected_indices.append(best_idx)
            available_indices.remove(best_idx)
            total_score += user_scores[best_idx]
            total_local_count += (user_countries[best_idx] == user_country_code)
            total_us_count += (user_countries[best_idx] == us_index)

        return selected_indices

        

    # Run the efficient version for local and global
    bf_scores_mixed_efficient1 = {}
    bf_scores_mixed_efficient2 = {}
    bf_scores_mixed_efficient3 = {}

    for user in tqdm.tqdm(scores1.index):
        bf_scores_mixed_efficient1[user] = efficient_brute_force_local_and_global(scores1, user, countries_candidates1, k, l=args.l, bias_local_weight=args.bias_local_weight)
        bf_scores_mixed_efficient2[user] = efficient_brute_force_local_and_global(scores2, user, countries_candidates2, k, l=args.l, bias_local_weight=args.bias_local_weight)
        bf_scores_mixed_efficient3[user] = efficient_brute_force_local_and_global(scores3, user, countries_candidates3, k, l=args.l, bias_local_weight=args.bias_local_weight)

    bf_scores_mixed_efficient1 = pd.DataFrame(bf_scores_mixed_efficient1)
    bf_scores_mixed_efficient2 = pd.DataFrame(bf_scores_mixed_efficient2)
    bf_scores_mixed_efficient3 = pd.DataFrame(bf_scores_mixed_efficient3)

    # Save results
    bf_scores_mixed_efficient1.to_csv(args.path_output + "bf_scores_mixed1.csv", index=False)
    bf_scores_mixed_efficient2.to_csv(args.path_output + "bf_scores_mixed2.csv", index=False)
    bf_scores_mixed_efficient3.to_csv(args.path_output + "bf_scores_mixed3.csv", index=False)
    print("Results saved to", args.path_output)

if __name__ == '__main__':
    main()