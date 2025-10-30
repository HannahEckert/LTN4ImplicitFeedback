from models import MatrixFactorization, MFTrainer, MultVAE, TrainerMultVAE
from loaders import TrainingDataLoader, ValidationDataLoader, TrainingDataLoaderMultVAE, ValidationDataLoaderMultVAE
import torch
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard


def parse_args():
    parser = argparse.ArgumentParser(description='Neural Matrix Factorization for Recommendation System')

    # seed
    parser.add_argument('--seed', type=int, default=44,
                        help='Random seed')
    
    # Data paths
    parser.add_argument("--data_folder", type=str, default = "data_splits/")
    parser.add_argument("--user_country_mapping_file", type=str, default="results_full_sample/country_data/user_country_mapping.csv")
    parser.add_argument("--model_save_path", type=str, default="model.pth",
                       help="Path to save the trained model")
    parser.add_argument("--val_save_path", type=str, default="validation_data.pth",
                       help="Path to save the validation data")
    parser.add_argument("--results_save_path", type=str, default="results.pth",
                       help="Path to save the results of the validation")
    
    # Model parameters
    parser.add_argument('--n_factors', type=int, default=64,
                       help='Number of latent factors in the matrix factorization')
    parser.add_argument("--loss_function", type=str, default="bpr",
                       help="Loss function to use: bpr or bce")
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay (L2 penalty) for optimizer')
    
    # Validation parameters
    parser.add_argument('--validation_metric', type=str, default='ndcg@10',
                       help='Validation metric to use')
    parser.add_argument('--validation_freq', type=int, default=3,
                       help='Frequency of validation (in epochs)')
    parser.add_argument('--use_validation', action='store_true',
                       help='Use validation set')
    
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=3,
                       help='Number of epochs to wait before early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001,
                       help='Minimum change in validation metric to qualify as improvement')
    
    # GPU parameters
    parser.add_argument('--gpu', type=int, default=-1,
                        help="Which GPU to use (set to -1 for CPU, or specify GPU index)")
    
    # TensorBoard logging
    parser.add_argument('--log_dir', type=str, default='runs',
                        help="Directory to save TensorBoard logs")
    

    
    return parser.parse_args()


def main():
    args = parse_args()

    #print arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Set random seeds 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Set device based on GPU availability
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device('cpu')  # Use CPU
        print("Using CPU")
    else:
        device = torch.device(f'cuda:{args.gpu}')  # Use specified GPU
        print(f"Using GPU: {device}")

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(args.log_dir)

    all_user_items = torch.load(Path(args.data_folder) / "all_user_items.pth", weights_only=False)
    user_features = torch.load(Path(args.data_folder) / "user_features.pth", weights_only=False)
    item_features = torch.load(Path(args.data_folder) / "item_features.pth", weights_only=False)
    data_test = np.load(Path(args.data_folder) / "data_test.npy")
    data_validation = np.load(Path(args.data_folder) / "data_validation.npy")
    data_train = np.load(Path(args.data_folder) / "data_train.npy")

    all_user_items_train = torch.load(Path(args.data_folder) / "all_user_items_train.pth", weights_only=False)

    n_users = len(user_features)
    n_items = len(item_features)

    print(f"Train size: {len(data_train)}, Test size: {len(data_test)}, Validation size: {len(data_validation)}")

    bpr = args.loss_function == "bpr"
    model = MatrixFactorization(n_users, n_items, 
                                    n_factors=args.n_factors, 
                                    bpr=bpr).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate, 
                                weight_decay=args.weight_decay)

    trainer = MFTrainer(model, optimizer, item_features)

    train_loader = TrainingDataLoader(data_train, user_features, 
                                    item_features, n_items=n_items, 
                                    batch_size=args.batch_size)
    if args.use_validation:
        test_loader_metric = ValidationDataLoader(data_validation, user_features, 
                                                item_features, all_user_items, 
                                                n_items, batch_size=args.batch_size)
        test_loader_loss = TrainingDataLoader(data_validation, user_features, 
                                                item_features, n_items=n_items, 
                                                batch_size=args.batch_size)
    
    # Early stopping variables
    best_metric = -np.inf  # For metrics like NDCG where higher is better
    epochs_without_improvement = 0
    best_model_state = None

    # Train the model
    for epoch in range(args.n_epochs):
        if args.architecture == "NeuMF":
            train_loss = trainer.train_epoch(train_loader, loss_function=args.loss_function)
            test_loss = trainer.compute_test_loss(test_loader_loss, loss_function=args.loss_function)
        elif args.architecture == "MultVAE":
            train_loss = trainer.train_epoch(train_loader,epoch=epoch,total_epochs=args.n_epochs)
        else:
            raise ValueError(f"Unknown architecture: {args.architecture}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)

        if epoch % args.validation_freq == 0:
            ndcg = trainer.validate(test_loader_metric, args.validation_metric)
            print(f"Epoch {epoch+1}/{args.n_epochs} - Training Loss: {train_loss:.4f} - Test Loss {test_loss:.4f} - {args.validation_metric}: {ndcg:.4f}")

            # Log validation metrics
            writer.add_scalar(f'Validation/{args.validation_metric}', ndcg, epoch)
            
            # Early stopping check
            if args.early_stopping:
                if ndcg > best_metric + args.min_delta:
                    best_metric = ndcg
                    epochs_without_improvement = 0
                    best_model_state = model.state_dict().copy()
                    print(f"New best {args.validation_metric}: {best_metric:.4f}")
                else:
                    epochs_without_improvement += args.validation_freq
                    print(f"No improvement for {epochs_without_improvement} epochs")
                    
                    if epochs_without_improvement >= args.patience:
                        print(f"Early stopping after {epoch+1} epochs. Best {args.validation_metric}: {best_metric:.4f}")
                        # Restore best model
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                        break
        else:
            print(f"Epoch {epoch+1}/{args.n_epochs} - Training Loss: {train_loss:.4f} - Test Loss {test_loss:.4f}")
    # Final validation
    if args.early_stopping and best_model_state is not None:
        # Use best metric from early stopping
        final_score = best_metric
        print(f"Using best model from early stopping with {args.validation_metric}: {final_score:.4f}")
    else:
        # Run final validation
        results = trainer.validate(test_loader_metric, args.validation_metric, give_userwise=True)
        final_score = ndcg if 'ndcg' in locals() else results

    # Get final results for saving
    results = trainer.validate(test_loader_metric, args.validation_metric, give_userwise=True)

    # Save the model
    torch.save(model.state_dict(), args.model_save_path)
    # save validation data
    torch.save(data_validation, args.val_save_path)
    # Save results
    torch.save(results, args.results_save_path)



    # Close the TensorBoard writer
    writer.close()

    print(f"Validation score: {final_score:.4f}") 


if __name__ == '__main__':
    main()

