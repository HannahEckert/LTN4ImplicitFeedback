'''
code partly from https://github.com/tommasocarraro/LTNrec
'''

import ltn
import torch
import torch.nn as nn
import numpy as np
from metrics import compute_metric, check_metrics
import torch.nn.functional as F

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors, bpr=False):
        """
        If bpr = False
        Hybrid model: Matrix Factorization (MF) + Multi-Layer Perceptron (MLP) with bce loss
        If bpr = True
        Matrix Factorization (MF) model optimized with Bayesian Personalized Ranking (BPR) loss
        """
        super(MatrixFactorization, self).__init__()

        # User and item embeddings for MF branch
        self.u_emb = nn.Embedding(n_users, n_factors)
        self.i_emb = nn.Embedding(n_items, n_factors)
        self.bpr = bpr

        # Numerical feature (age) transformation
        self.u_age_fc = nn.Linear(1, n_factors // 2)

        # MLP to learn interactions from explicit features
        mlp_input_dim = (
            n_factors  # User embedding
            + n_factors  # Item embedding
        )

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_input_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Output a single prediction score
        )


        # Initialize embeddings
        nn.init.xavier_normal_(self.u_emb.weight)
        nn.init.xavier_normal_(self.i_emb.weight)

    #def forward(self, u_idx, i_idx, u_feat, i_feat):
    def forward(self, user, item):

        u_idx = user[:, 0].long()
        i_idx = item[:, 0].long()


        # MF part: Latent embeddings dot product
        u_emb = self.u_emb(u_idx)
        i_emb = self.i_emb(i_idx)

        # for embedings that come from LTN variables
        if len(u_emb.shape) == 3:
            u_emb = u_emb.squeeze()
        if len(i_emb.shape) == 3:
            i_emb = i_emb.squeeze()

        mf_pred = torch.sum(u_emb * i_emb, dim=1)

        

        user_features = u_emb
        item_features = i_emb
        mlp_input = torch.cat([user_features, item_features], dim=1)

        # MLP output
        mlp_pred = self.mlp(mlp_input).squeeze()

        if self.bpr:
            pred = mf_pred
        else:
            # Final prediction: MF + MLP
            pred = mf_pred + mlp_pred


        if self.bpr:
            return pred
        else:
            return torch.sigmoid(pred)  # Output in range [0,1]
        




class Trainer:
    """
    Abstract base class that any trainer must inherit from.
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, train_loader, val_loader, val_metric, n_epochs=200, early=None, verbose=10, save_path=None):
        """
        Method for the train of the model.

        :param train_loader: data loader for training data
        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :param n_epochs: number of epochs of training, default to 200
        :param early: threshold for early stopping, default to None
        :param verbose: number of epochs to wait for printing training details (every 'verbose' epochs)
        :param save_path: path where to save the best model, default to None
        """
        best_val_score = 0.0
        early_counter = 0
        check_metrics(val_metric)

        for epoch in range(n_epochs):
            # training step
            train_loss = self.train_epoch(train_loader)
            # validation step
            val_score = self.validate(val_loader, val_metric)
            # print epoch data
            if (epoch + 1) % verbose == 0:
                print("Epoch %d - Train loss %.3f - Validation %s %.3f"
                      % (epoch + 1, train_loss, val_metric, val_score))
            # save best model and update early stop counter, if necessary
            if val_score > best_val_score:
                best_val_score = val_score
                early_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                early_counter += 1
                if early is not None and early_counter > early:
                    print("Training interrupted due to early stopping")
                    break

    def train_epoch(self, train_loader):
        """
        Method for the training of one single epoch.

        :param train_loader: data loader for training data
        :return: training loss value averaged across training batches
        """
        raise NotImplementedError()

    def predict(self, x, *args, **kwargs):
        """
        Method for performing a prediction of the model.

        :param x: input for which the prediction has to be performed
        :param args: these are the potential additional parameters useful to the model for performing the prediction
        :param kwargs: these are the potential additional parameters useful to the model for performing the prediction
        :return: prediction of the model for the given input
        """

    def validate(self, val_loader, val_metric):
        """
        Method for validating the model.

        :param val_loader: data loader for validation data
        :param val_metric: validation metric name
        :return: validation score based on the given metric averaged across validation batches
        """
        raise NotImplementedError()

    def save_model(self, path):
        """
        Method for saving the model.

        :param path: path where to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        """
        Method for loading the model.

        :param path: path from which the model has to be loaded.
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def test(self, test_loader, metrics):
        """
        Method for performing the test of the model based on the given test data and test metrics.

        :param test_loader: data loader for test data
        :param metrics: metric name or list of metrics' names that have to be computed
        :return: a dictionary containing the value of each metric average across the test batches
        """
        raise NotImplementedError()
    

class MFTrainer(Trainer):
    def __init__(self, mf_model, optimizer, item_features):
        self.item_features = item_features
        super().__init__(mf_model, optimizer)
    
   
    #define bpr loss
    def bpr_loss(self, pos_pred, neg_pred):
        if pos_pred is None or neg_pred is None:
            raise ValueError("BPR loss received None predictions")

        return -torch.mean(F.logsigmoid(pos_pred.squeeze() - neg_pred.squeeze()))
    
    #bce loss
    bce_loss = torch.nn.BCELoss()
    


    def train_epoch(self, train_loader, loss_function="bce", factor_axiom_vs_task=1):
        train_loss = 0.0
        for batch_idx, (user, positive_item, negative_item) in enumerate(train_loader):
            self.optimizer.zero_grad()

            # Move all inputs to the same device as the model
            device = next(self.model.parameters()).device
            user = user.to(device)
            positive_item = positive_item.to(device)
            negative_item = negative_item.to(device)

            pos_pred = self.model(user, positive_item)
            neg_pred = self.model(user, negative_item)

            if loss_function == "bpr":
                loss = self.bpr_loss(pos_pred, neg_pred)
            elif loss_function == "bce":
                # Move targets to the same device as predictions
                pos_target = torch.ones_like(pos_pred, device=device)
                neg_target = torch.zeros_like(neg_pred, device=device)
                loss = self.bce_loss(pos_pred, pos_target) + self.bce_loss(neg_pred, neg_target)
            
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss / len(train_loader)
    
    def validate(self, val_loader, val_metric, give_userwise = False, give_everything = False):
        """
        Validation logic using NDCG/Recall/Hit.
        Optimized for GPU usage and batch processing.
        
        :param val_loader: validation data loader
        :param val_metric: validation metric, e.g., 'ndcg@10'
        """
        self.model.eval()
        device = next(self.model.parameters()).device  # Get device once
        
        all_scores = []
        all_ground_truth = []
        all_candidates = []

        with torch.no_grad():
            for user, positive_item, negative_items in val_loader:
                # Move entire batch to device at once
                user = user.to(device)
                positive_item = positive_item.to(device)
                negative_items = negative_items.to(device)
                
                batch_size = user.size(0)
                n_candidates = 1 + negative_items.size(1)  # 1 positive + n negatives
                
                # Combine all candidates (positive first, then negatives)
                candidates = torch.cat([
                    positive_item.unsqueeze(1),  # [batch_size, 1, 1 + item_feat_dim]
                    negative_items
                ], dim=1)  # shape: [batch_size, n_candidates, 1 + item_feat_dim]
                
                # Prepare user tensor for batch processing
                user_repeated = user.unsqueeze(1).expand(-1, n_candidates, -1)
                
                # Reshape for batch processing
                user_flat = user_repeated.reshape(-1, user_repeated.size(-1))
                candidates_flat = candidates.reshape(-1, candidates.size(-1))
                
                # Compute all scores in one forward pass
                scores_flat = self.model(user_flat, candidates_flat)
                scores = scores_flat.reshape(batch_size, n_candidates)
                
                # Create ground truth matrix
                ground_truth = torch.zeros(batch_size, n_candidates, device=device)
                ground_truth[:, 0] = 1  # First item is positive
                
                all_scores.append(scores.cpu())
                all_ground_truth.append(ground_truth.cpu())
                all_candidates.append(candidates.cpu())

        # Concatenate all batches using torch (faster than numpy for GPU tensors)
        all_scores = torch.cat(all_scores, dim=0).numpy()
        all_ground_truth = torch.cat(all_ground_truth, dim=0).numpy()
        all_candidates = torch.cat(all_candidates, dim=0).numpy()

        # Compute metric
        result = compute_metric(val_metric, all_scores, all_ground_truth)

        if give_everything:
            return result, all_candidates, all_scores

        if give_userwise:
            return result
        else:
            return np.mean(result)  # Average over users
    


    
    def compute_test_loss(self, test_loader, loss_function="bce", give_userwise=False):
        """
        Compute average test loss without updating gradients.
        """
        self.model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        userwise_loss = []
        users = []

        with torch.no_grad():  # No gradients for test
            for batch_idx, (user, positive_item, negative_item) in enumerate(test_loader):

                # Move all inputs to the same device as the model
                device = next(self.model.parameters()).device
                user = user.to(device)
                positive_item = positive_item.to(device)
                negative_item = negative_item.to(device)

                pos_pred = self.model(user, positive_item)
                neg_pred = self.model(user, negative_item)

                if loss_function == "bpr":
                    loss = self.bpr_loss(pos_pred, neg_pred)
                elif loss_function == "bce":
                    loss = self.bce_loss(pos_pred, torch.ones_like(pos_pred)) + self.bce_loss(neg_pred, torch.zeros_like(neg_pred))

                test_loss += loss.item()
                if give_userwise:
                    userwise_loss.append(loss)
                    users.append(user.cpu().numpy())  # Store user IDs for userwise loss
        if give_userwise:
            return np.array(users), torch.tensor(userwise_loss)
        else:
            return test_loss / len(test_loader)
 

class LTNTrainerMF(MFTrainer):
    """
    Trainer for the Logic Tensor Network with NeuMF as predictive model for the Likes function

    The Likes function takes as input a user-item pair and produce an un-normalized score (MF). Ideally, this score
    should be near 1 if the user likes the item, and near 0 if the user dislikes the item.

    """
    
    def __init__(self, mf_model, optimizer, item_features, M_user = [], M_item = [],user_countries = [], Mainstreaminess_axioms = False, f1 = False, f2 = False, alpha = 1):
        """
        Constructor of the trainer for the LTN with MF as base model.

        :param mf_model: Matrix Factorization model to implement the Likes function
        :param optimizer: optimizer used for the training of the model
        :param item_features: dictionary {item_id: feature_vector}
        :param M_user must be df of the form [user_id, global M, local M]
        :param M_item must be of the form [global, country1, .. , countryN]
        :param user_countries: dataframe of the form [user_id, user_country]
        """
    
        super().__init__(mf_model, optimizer, item_features)

        self.device = next(self.model.parameters()).device
        print("using:", self.device)

        self.item_features = item_features
        self.Mainstreaminess_axioms = Mainstreaminess_axioms
        self.f1 = f1
        self.f2 = f2
        self.Likes = ltn.Function(mf_model)
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier='f')
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.sat_agg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=1))
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

        #for bpr
        self.BigDiff = ltn.Predicate(func=lambda pred_pos_example, pred_neg_example: torch.sigmoid(pred_pos_example - pred_neg_example))

        if mf_model.bpr:
            self.GoodPosPred = ltn.Predicate(func=lambda pred: torch.sigmoid(alpha * pred))
            self.GoodNegPred = ltn.Predicate(func=lambda pred: 1 - torch.sigmoid(alpha * pred))
        else:
            #for bce
            self.GoodPosPred = ltn.Predicate(func=lambda pred: pred)
            self.GoodNegPred = ltn.Predicate(func=lambda pred: 1 - pred)

        #for mainstreaminess
        if list(M_user) and list(M_item) and list(user_countries):

            # Precompute these once during initialization
            self.user_g_dict = dict(zip(M_user.iloc[:, 0], M_user.iloc[:, 1]))
            self.user_l_dict = dict(zip(M_user.iloc[:, 0], M_user.iloc[:, 2]))
            self.item_g_dict = dict(zip(M_item.index, M_item.iloc[:, 0]))

            # Create user-country lookup
            user_country_dict = dict(zip(user_countries.iloc[:, 0], user_countries.iloc[:, 1]))

            # Create item-country matrix lookup (assuming M_item has country columns)
            self.item_country_dict = {}
            for item_idx, row in M_item.iterrows():
                self.item_country_dict[item_idx] = row.to_dict()

            # Then define predicates
            self.M_G_user = ltn.Predicate(func=lambda user: torch.tensor(
                [self.user_g_dict[int(u)] for u in user[:, 0]], 
                device=self.device,
                dtype=torch.float32
            ))

            self.M_L_user = ltn.Predicate(func=lambda user: torch.tensor(
                [self.user_l_dict[int(u)] for u in user[:, 0]],
                device=self.device,
                dtype=torch.float32
            ))

            self.M_G_item = ltn.Predicate(func=lambda item: torch.tensor(
                [self.item_g_dict[int(i)] for i in item[:, 1]],
                device=self.device,
                dtype=torch.float32
            ))
             # Precompute user-country mapping as GPU tensor
            user_country_dict = dict(zip(user_countries.iloc[:, 0], user_countries.iloc[:, 1]))
            max_user_id = max(user_country_dict.keys())
            
            # Map countries to indices (assuming M_item columns are the countries)
            country_columns = M_item.columns.tolist()
            self.country_to_idx = {country: idx for idx, country in enumerate(country_columns)}
            
            # Create user-country index tensor
            self.user_country_indices = torch.zeros(max_user_id + 1, dtype=torch.long, device=self.device)
            for user_id, country in user_country_dict.items():
                self.user_country_indices[user_id] = self.country_to_idx.get(country, 0)
            
            # Create item-country matrix as GPU tensor
            self.item_country_matrix = torch.tensor(
                M_item.values, device=self.device, dtype=torch.float32
            )
            
            # Redefine M_L_item with vectorized GPU operations
            self.M_L_item = ltn.Predicate(func=lambda item, user: 
                self.item_country_matrix[item[:, 1].long(), self.user_country_indices[user[:, 0].long()]]
            )

    def train_epoch(self, train_loader, loss_function="bce", factor_axiom_vs_task=1, factor_f2 = 10):
        train_loss = 0.0
        #for batch_idx, (u_idx, i_pos_idx, i_neg_idx, u_feat, i_pos_feat, i_neg_feat) in enumerate(train_loader):

        for batch_idx, (data) in enumerate(train_loader):

            if train_loader.f1 and train_loader.f2:
               user_f1, positive_item_f1, negative_item_f1, user_f2, positive_item_f2, negative_item_f2, user, positive_item, negative_item = data

            elif train_loader.f1 or train_loader.f2 or train_loader.f3:
                user_axiom, positive_item_axiom, negative_item_axiom, user, positive_item, negative_item = data
                user_f1, positive_item_f1, negative_item_f1 = user_axiom, positive_item_axiom, negative_item_axiom
                user_f2, positive_item_f2, negative_item_f2 = user_axiom, positive_item_axiom, negative_item_axiom
            else:
                user, positive_item, negative_item = data

            self.optimizer.zero_grad()


            if loss_function == "bce":

                train_sat_pos = self.Forall(ltn.diag(user, positive_item),
                                        self.GoodPosPred(self.Likes(user, positive_item))).value
                train_sat_neg = self.Forall(ltn.diag(user, negative_item),
                                        self.GoodNegPred(self.Likes(user, negative_item))).value
                
                

                train_sat = self.sat_agg(train_sat_pos, train_sat_neg)

            elif loss_function == "bpr":

                train_sat = self.Forall(ltn.diag(user, positive_item, negative_item),
                                        self.BigDiff(self.Likes(user,positive_item),self.Likes(user,negative_item))).value
                


            if self.Mainstreaminess_axioms and batch_idx < len(train_loader) -1:

                try:
                    
                    if self.f1:
                        f1 = self.Forall(ltn.undiag(user_f1, negative_item_f1),
                                            self.Implies(
                                                self.And(self.Not(self.M_G_user(user_f1)),self.M_G_item(negative_item_f1)),
                                                self.GoodNegPred(self.Likes(user_f1, negative_item_f1))
                                            )).value

                    if self.f2:

                        f2 = self.Forall(ltn.undiag(user_f2, negative_item_f2),
                                            self.Implies(
                                                self.And(self.M_L_user(user_f2),self.M_L_item(negative_item_f2, user_f2)),
                                                self.GoodPosPred(self.Likes(user_f2, negative_item_f2))
                                            )).value

                    if not (self.f1 or self.f2):
                        raise ValueError("At least one of f1, f2 must be True for mainstreaminess axioms.")

                    # Combine only the defined axioms
                    if self.f1 and self.f2:
                        if factor_f2 == 10:
                            train_sat_M_axioms = self.sat_agg(f1, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2) #weigh f2 more because first part of implication is less likely
                        elif factor_f2 == 20:
                            train_sat_M_axioms = self.sat_agg(f1, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2, f2) #for bpr even more because US overrepresentation is less prominent
                        elif factor_f2 == 100:
                            sat_agg_p1 = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=1))
                            tempf1  = sat_agg_p1(f1, f2, f2, f2, f2, f2, f2, f2, f2, f2)
                            tempf2 = sat_agg_p1(f2, f2, f2, f2, f2, f2, f2, f2, f2, f2)
                            train_sat_M_axioms = self.sat_agg(tempf1, tempf2,tempf2, tempf2, tempf2, tempf2, tempf2, tempf2, tempf2, tempf2)
                        elif factor_f2 == 1000:
                            sat_agg_p1 = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=1))
                            tempf1_10  = sat_agg_p1(f1, f2, f2, f2, f2, f2, f2, f2, f2, f2)
                            tempf1 = sat_agg_p1( tempf1_10, f2, f2, f2, f2, f2, f2, f2, f2, f2)
                            tempf2 = sat_agg_p1(f2, f2, f2, f2, f2, f2, f2, f2, f2, f2)
        
                            train_sat_M_axioms = self.sat_agg(tempf1, tempf2,tempf2, tempf2, tempf2, tempf2, tempf2, tempf2, tempf2, tempf2)
                        else:
                            raise ValueError("factor_f2 must be 10, 20 or 100.")
                    elif self.f1:
                        train_sat_M_axioms = f1
                    elif self.f2:
                        train_sat_M_axioms = f2
                    else:
                        raise ValueError("At least one of f1, f2 must be True for mainstreaminess axioms.")

                    if factor_axiom_vs_task == 1:
                        train_sat = self.sat_agg(train_sat, train_sat_M_axioms)

                    elif factor_axiom_vs_task ==5:
                        train_sat = self.sat_agg(train_sat, train_sat, train_sat, train_sat, train_sat, train_sat_M_axioms)

                    elif factor_axiom_vs_task ==10:
                        train_sat = self.sat_agg(train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat_M_axioms)

                    elif factor_axiom_vs_task == 20:
                        train_sat = self.sat_agg(train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat_M_axioms)
                    elif factor_axiom_vs_task == 40:
                        train_sat = self.sat_agg(train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat_M_axioms)
                    elif factor_axiom_vs_task == 100:
                        sat_agg_p1 = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregPMeanError(p=1))
                        temp  = sat_agg_p1(train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat, train_sat_M_axioms)
                        train_sat = self.sat_agg(train_sat, train_sat,train_sat, train_sat,train_sat, train_sat,train_sat, train_sat,train_sat, train_sat, temp)


                    else:
                        raise ValueError("factor_axiom_vs_task must be 1, 5, 10, 100, 40 or 20.")



                except Exception as e:
                    if e == ValueError:
                        raise e
                    else:
                        pass

            loss = 1. - train_sat
            loss.backward()
            self.optimizer.step()
            train_loss += train_sat.item()


        return train_loss / len(train_loader)
    
    def compute_test_loss(self, test_loader, loss_function="bce"):
        """
        Compute average test loss without updating gradients.
        """
        self.model.eval()
        test_loss = 0.0
        test_loss_axioms = 0.0
        test_loss_task = 0.0

        with torch.no_grad():

            for batch_idx, (user, positive_item, negative_item) in enumerate(test_loader):

                if loss_function == "bce":

                    test_sat_pos = self.Forall(ltn.diag(user, positive_item),
                                            self.GoodPosPred(self.Likes(user, positive_item))).value
                    test_sat_neg = self.Forall(ltn.diag(user, negative_item),
                                            self.GoodNegPred(self.Likes(user, negative_item))).value

                    test_sat = self.sat_agg(test_sat_pos, test_sat_neg)

                elif loss_function == "bpr":

                    test_sat = self.Forall(ltn.diag(user, positive_item, negative_item),
                                            self.BigDiff(self.Likes(user,positive_item),self.Likes(user,negative_item))).value


                if self.Mainstreaminess_axioms:

                    try:

                        if self.f1:
                            f1 = self.Forall(ltn.diag(user, negative_item),
                                                self.Implies(
                                                    self.And(self.Not(self.M_G_user(user)),self.M_G_item(negative_item)),
                                                    self.GoodNegPred(self.Likes(user, negative_item))
                                                )).value

                        if self.f2:
                            f2 = self.Forall(ltn.diag(user, positive_item),
                                                self.Implies(
                                                    self.And(self.M_L_user(user),self.M_L_item(positive_item, user)),
                                                    self.GoodPosPred(self.Likes(user, positive_item))
                                                )).value
                            
                        if not (self.f1 or self.f2 or self.f3):
                            raise ValueError("At least one of f1, f2, f3 must be True for mainstreaminess axioms.")

                        if self.f1 and self.f2:
                            test_sat_M_axioms = self.sat_agg(f1, f2)
                        elif self.f1:
                            test_sat_M_axioms = f1
                        elif self.f2:
                            test_sat_M_axioms = f2
                        else:
                            raise ValueError("At least one of f1, f2 must be True for mainstreaminess axioms.")

                        test_loss_task += test_sat.item()
                        test_sat = self.sat_agg(test_sat, test_sat_M_axioms)
                        
                        test_loss_axioms += test_sat_M_axioms.item()
                    except Exception as e:
                        pass

                test_loss += test_sat.item()

        if self.Mainstreaminess_axioms:
            return test_loss / len(test_loader), test_loss_axioms / len(test_loader), test_loss_task / len(test_loader)
        
        else:
            return test_loss / len(test_loader)
        
    
    


class TrainerMultVAE(Trainer):
    """
    Trainer for the MultVAE model.
    """
    def __init__(self, vae_model, optimizer, beta=1.0):
        super(TrainerMultVAE, self).__init__(vae_model, optimizer)
        self.vae_model = vae_model
        self.optimizer = optimizer
        self.beta = beta #max value of beta for KL annealing

    def kl_anneal(self, epoch, total_epochs):
        """
        KL Annealing: Gradually increase the weight of the KL divergence term.
        Linear annealing from beta=0 to beta=1 over training epochs.
        """
        return self.beta * (epoch / total_epochs)
    
    def loss_function(self, logits, target, mu, logvar, epoch, total_epochs):
        """
        Loss function for the model.
        :param logits: predicted logits
        :param target: target interactions
        :param mu: mean of the latent space
        :param logvar: log variance of the latent space
        :param epoch: current epoch
        :param total_epochs: total number of epochs
        """
        # KL annealing
        beta = self.kl_anneal(epoch, total_epochs)

        # Multinomial NLL loss
        log_softmax_logits = F.log_softmax(logits, dim=1)
        nll = -torch.sum(target * log_softmax_logits, dim=1).mean()

        # KL divergence
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        loss = nll + beta * kl

        return loss
    
    def train_epoch(self, train_loader, epoch, total_epochs):
        """
        Train the model for one epoch.
        :param train_loader: data loader for training data
        :param epoch: current epoch
        :param total_epochs: total number of epochs
        """

        train_loss = 0.0
        self.model.train()

        for batch_idx, (user, interaction) in enumerate(train_loader):
            self.optimizer.zero_grad()

            # Move all inputs to the same device as the model
            device = next(self.vae_model.parameters()).device
            interaction = interaction.to(device)
            user = user.to(device)
            user = user.long()

            # Forward pass
            logits, mu, logvar = self.vae_model(interactions=interaction, user=user)

            loss = self.loss_function(logits, interaction, mu, logvar, epoch, total_epochs)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss / len(train_loader)
    
    def validate(self, val_loader, val_metric, give_userwise=False, give_everything=False):
        """
        Validation logic using NDCG/Recall/Hit.
        Optimized for GPU usage and batch processing.
        
        :param val_loader: validation data loader
        :param val_metric: validation metric, e.g., 'ndcg@10'
        """
        self.model.eval()
        device = next(self.model.parameters()).device  # Get device once
        
        all_scores = []
        all_ground_truth = []
        all_candidates = []

        with torch.no_grad():
            for u_idx, u_feats, candidates, user_interactions in val_loader:

                # Move entire batch to device at once
                u_idx = u_idx.to(device)
                u_feats = u_feats.to(device)
                candidates = candidates.to(device)
                user_interactions = user_interactions.to(device)

                batch_size = u_idx.size(0)
                n_candidates = candidates.size(1)

                u_feats = u_feats.long()


                logits, mu, logvar = self.vae_model(interactions=user_interactions, user=u_feats)

                scores = torch.stack([logits[ind][candidates[ind]] for ind in range(batch_size)])

                # Create ground truth matrix
                ground_truth = torch.zeros(batch_size, n_candidates, device=device)
                ground_truth[:, 0] = 1  # First item is positive

                all_scores.append(scores.cpu())
                all_ground_truth.append(ground_truth.cpu())
                all_candidates.append(candidates.cpu())
        # Concatenate all batches using torch (faster than numpy for GPU tensors)
        all_scores = torch.cat(all_scores, dim=0).numpy()
        all_ground_truth = torch.cat(all_ground_truth, dim=0).numpy()
        all_candidates = torch.cat(all_candidates, dim=0).numpy()

        # Compute metric
        result = compute_metric(val_metric, all_scores, all_ground_truth)

        if give_everything:
            return result, all_candidates, all_scores

        if give_userwise:
            return result
        else:
            return np.mean(result)  # Average over users

        

    
    def compute_test_loss(self, test_loader, epoch, total_epochs):
        """
        Compute average test loss without updating gradients.
        """
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (user, interaction) in enumerate(test_loader):
                # Move all inputs to the same device as the model
                device = next(self.vae_model.parameters()).device
                interaction = interaction.to(device)
                user = user.to(device)
                user = user.long()

                # Forward pass
                logits, mu, logvar = self.vae_model(interaction, user)

                loss = self.loss_function(logits, interaction, mu, logvar, epoch, total_epochs)
                test_loss += loss.item()
        return test_loss / len(test_loader)
    





