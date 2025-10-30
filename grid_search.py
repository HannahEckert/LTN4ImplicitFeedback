import itertools
import subprocess
import csv
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Matrix Factorization for Recommendation System')

    #lists of parameters to try
    parser.add_argument("--n_factors", type=lambda s: [int(x) for x in s.split(',')], default=["64", "128"],
                        help="List of number of latent factors to try")
    parser.add_argument("--loss_function", type=lambda s: [str(x) for x in s.split(',')], default=["bpr", "bce"],
                        help="List of loss functions to try")
    parser.add_argument("--batch_size", type=lambda s: [int(x) for x in s.split(',')], default=["1024", "2048"],
                        help="List of batch sizes to try")
    parser.add_argument("--learning_rate", type=lambda s: [float(x) for x in s.split(',')], default=["0.001", "0.01"],
                        help="List of learning rates to try")
    parser.add_argument("--weight_decay", type=lambda s: [float(x) for x in s.split(',')], default=["0.0", "1e-4"],
                        help="List of weight decay values to try")
    
    #fixed parameters
    parser.add_argument("--train_file", type=str, default="train_LTN.py",
                        help="Path to the training file")
    parser.add_argument("--grid_search_save_path", type=str, default="grid_search_results.csv",
                        help="Path to save the grid search results")
    parser.add_argument("--data_folder", type=str, default = "data_splits/")
    parser.add_argument("--user_country_mapping_file", type=str, default="country_data/user_country_mapping.csv")
    parser.add_argument("--model_save_path", type=str, default="model.pth",
                       help="Path to save the trained model")
    parser.add_argument('--n_epochs', type=int, default=11,
                       help='Number of training epochs')
    parser.add_argument('--validation_metric', type=str, default='ndcg@10',
                       help='Validation metric to use')
    parser.add_argument('--validation_freq', type=int, default=10,
                       help='Frequency of validation (in epochs)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help="Which GPU to use (set to -1 for CPU, or specify GPU index)")
    
    #mainstreaminess 
    parser.add_argument("--M_user", type=str, default="mainstreaminess_measures/distribution_based_user_M.csv",
                        help="Path to the user mainstreaminess measures file")
    parser.add_argument("--mainstreaminess_axioms", type=bool, default=False,
                        help="Use mainstreaminess axioms")
    parser.add_argument("--f1", type=bool, default=False)
    parser.add_argument("--f2", type=bool, default=False)


    return parser.parse_args()

def main():
    args = parse_args()

    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    param_grid = {
        'n_factors': args.n_factors,
        'loss_function': args.loss_function,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    #Log results

    with open(args.grid_search_save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(keys) + ["result"])

        for combination in combinations:
            print("Running with parameters:", combination)
            command = ["python", args.train_file]
            for key, value in combination.items():
                command.append(f"--{key}")
                command.append(str(value))

            command.append("--data_folder")
            command.append(args.data_folder)
            command.append("--user_country_mapping_file")
            command.append(args.user_country_mapping_file)
            command.append("--model_save_path")
            command.append(args.model_save_path)
            command.append("--n_epochs")
            command.append(str(args.n_epochs))
            command.append("--validation_metric")
            command.append(args.validation_metric)
            command.append("--validation_freq")
            command.append(str(args.validation_freq))
            command.append("--gpu")
            command.append(str(args.gpu))

            if args.mainstreaminess_axioms:
                command.append("--mainstreaminess_axioms")
                command.append(str(args.mainstreaminess_axioms))
                command.append("--M_user")
                command.append(str(args.M_user))
                command.append("--M_artist")
                command.append(str(args.M_artist))
                if args.f1:
                    command.append("--f1")
                    command.append(str(args.f1))
                if args.f2:
                    command.append("--f2")
                    command.append(str(args.f2))

            #run command
            result = subprocess.run(command, capture_output=True, text=True)


            val_metric = "0.0"

            print("Command error:", result.stderr)

            if "Validation score" in result.stdout:
                val_metric = result.stdout.split("Validation score:")[1].strip()
                print("Validation score:", val_metric)
            else:
                print("No validation score found in output.")

            #save results
            writer.writerow(list(combination.values()) + [val_metric])

    print("Grid search completed. Results saved to ", args.grid_search_save_path)

            




if __name__ == '__main__':
    main()

