import pandas as pd
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim import AdamW
from torchsurv.loss import cox
import torchsurv
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.loss.cox import neg_partial_log_likelihood
from tqdm import tqdm
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
import torch.nn.functional as F
import random
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###################
## LOAD DATASETS
###################

for fold in range(1, 6):
    print(fold)
    #rna_train = pd.read_csv("~//Documents//COH Breast Cohort//Final Datasets//rna_fold1_train.csv")
    treat_train = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//treatment_fold{fold}_train.csv")
    #genomic_train = pd.read_csv("~//Documents//COH Breast Cohort//Final Datasets//genomic_fold5_train.csv")
    pathways_train = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//pathway_fold{fold}_train.csv")
    train_y = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//survival_group_fold{fold}_train.csv")
    
    train_x = pd.concat([pathways_train, treat_train], axis=1)
    
    #rna_val = pd.read_csv("~//Documents//COH Breast Cohort//Final Datasets//rna_fold5_val.csv")
    treat_val = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//treatment_fold{fold}_val.csv")
    #genomic_val = pd.read_csv("~//Documents//COH Breast Cohort//Final Datasets//genomic_fold5_val.csv")
    pathways_val = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//pathway_fold{fold}_val.csv")
    val_y = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//survival_group_fold{fold}_val.csv")
    
    val_x = pd.concat([pathways_val, treat_val], axis=1)
    
    #rna_test = pd.read_csv("~//Documents//COH Breast Cohort//Final Datasets//rna_fold5_test.csv")
    treat_test = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//treatment_fold{fold}_test.csv")
    #genomic_test = pd.read_csv("~//Documents//COH Breast Cohort//Final Datasets//genomic_fold5_test.csv")
    pathways_test = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//pathway_fold{fold}_test.csv")
    test_y = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//survival_group_fold{fold}_test.csv")
    
    test_x = pd.concat([pathways_test, treat_test], axis=1)
    
    #########################
    ## MIN-MAX NORMALIZATION
    #########################
    
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    
    xmin = np.amin(train_x)
    xmax = np.amax(train_x)
    train_x = (train_x - xmin) / (xmax - xmin)
    
    train_x = torch.from_numpy(train_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.float32))
    
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()
    
    test_x = (test_x - xmin) / (xmax - xmin)
    
    test_x = torch.from_numpy(test_x.astype(np.float32))
    test_y = torch.from_numpy(test_y.astype(np.float32))
    
    val_x = val_x.to_numpy()
    val_y = val_y.to_numpy()
    
    val_x = (val_x - xmin) / (xmax - xmin)
    
    val_x = torch.from_numpy(val_x.astype(np.float32))
    val_y = torch.from_numpy(val_y.astype(np.float32))
    
    
    #############################
    ## DEFINE MODEL ARCHITECTURE
    #############################

    class MultiTaskResponse(nn.Module):
        def __init__(
            self,
            input_dim,
            hidden_dim,
            hormone_extra=False,
            immune_extra=False,
            cdk_extra=False,
            chemo_extra=False
        ):
            super().__init__()
    
            self.hormone_extra = hormone_extra
            self.immune_extra = immune_extra
            self.cdk_extra = cdk_extra
            self.chemo_extra = chemo_extra
    
            feat_dim = hidden_dim // 4
    
            # Shared feature extractor
            self.combined = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
    
                nn.Linear(hidden_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            )
    
            # Optional feature normalization + activation layers per task
            def make_extra_block():
                return nn.Sequential(
                    nn.Linear(feat_dim, feat_dim),
                    nn.LayerNorm(feat_dim),
                    nn.ReLU()
                )
    
            self.hormone_block = make_extra_block() if hormone_extra else nn.Identity()
            self.immune_block  = make_extra_block() if immune_extra else nn.Identity()
            self.cdk_block     = make_extra_block() if cdk_extra else nn.Identity()
            self.chemo_block   = make_extra_block() if chemo_extra else nn.Identity()
    
            # Task heads
            def make_head():
                return nn.Sequential(
                    nn.Linear(feat_dim, hidden_dim // 8),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 8, hidden_dim // 16),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 16, 1)
                )
    
            self.pred_hormone = make_head()
            self.pred_immune  = make_head()
            self.pred_cdk     = make_head()
            self.pred_chemo   = make_head()
    
        def forward(self, x):
            feats = self.combined(x)
    
            hormone_out = self.pred_hormone(self.hormone_block(feats))
            immune_out  = self.pred_immune(self.immune_block(feats))
            cdk_out     = self.pred_cdk(self.cdk_block(feats))
            chemo_out   = self.pred_chemo(self.chemo_block(feats))
    
            return hormone_out, chemo_out, immune_out, cdk_out
    
    ###################################
    ## DEFINE EARLY STOPPING MECHANISM
    ###################################
    
    class EarlyStopping:
        """
        Early stopping to stop training when validation loss doesn't improve.
    
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Reload best model at the end
        """
        def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
            self.patience = patience
            self.min_delta = min_delta
            self.restore_best_weights = restore_best_weights
    
            self.best_loss = np.inf
            self.counter = 0
            self.best_state_dict = None
            self.early_stop = False
    
        def __call__(self, val_loss, model):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                if self.restore_best_weights:
                    self.best_state_dict = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.restore_best_weights and self.best_state_dict is not None:
                        model.load_state_dict(self.best_state_dict)
    
    
    ###################################
    ## DEFINE CUSTOM LOSS FUNCTION
    ###################################
    
    class MultiTaskLoss(nn.Module):
        def __init__(self):
            super().__init__()
    
            # Log-variance parameters (uncertainty weighting)
            # One per task
            self.log_vars = nn.Parameter(torch.zeros(10))
    
        def _task_loss(self, pred, event, time):
            """
            pred  : (N, 1)
            event : (N,)
            time  : (N,)
            """
            df = torch.cat(
                (pred, event.bool().unsqueeze(1), time.unsqueeze(1)), dim=1
            )
            mask = ~torch.any(torch.isnan(df), dim=1)
            df_filt = df[mask]
    
            if df_filt.shape[0] == 0:
                return None
    
            num_events = torch.sum(df_filt[:, 1])
            if num_events == 0:
                return None
    
            return neg_partial_log_likelihood(
                df_filt[:, 0],
                df_filt[:, 1].bool(),
                df_filt[:, 2]
            ) / num_events
    
        def forward(self, x1, x2, x3, x4, y):
    
            preds = [x1, x2, x3, x4]
    
            # (event_col, time_col) for each task
            label_map = [
                (1, 0),
                (7, 6),
                (5, 4),
                (3, 2),
            ]
    
            total_loss = 0.0
            active_tasks = 0
    
            for i, (pred, (e_col, t_col)) in enumerate(zip(preds, label_map)):
                loss_i = self._task_loss(
                    pred,
                    y[:, e_col],
                    y[:, t_col]
                )
    
                if loss_i is None:
                    continue
    
                # Uncertainty-weighted multitask loss
                total_loss += torch.exp(-self.log_vars[i]) * loss_i + self.log_vars[i]
                active_tasks += 1
    
            if active_tasks == 0:
                # No valid task in batch → return zero loss (safe)
                return total_loss
    
            return total_loss / active_tasks
    
    
    
    train_ci = []
    test_ci = []
    val_ci = []
    treat_list = []
    bs_list = []
    hd_list = []
    lr_list = []
    lr_cdk_list = []
    lr_chemo_list = []
    lr_hormone_list = []
    lr_immune_list = []
    lr_loss_list = []

    extra_cdk_list = []
    extra_chemo_list = []
    extra_hormone_list = []
    extra_immune_list = []
    
    
    for bs in [64]:
        for hd in [5000]:
            for lr in [0.00001, 0.00003]:
                for lr_h in [0.00003, 0.0001]:
                    for lr_cdk in [0.00003, 0.0001]:
                        for lr_chemo in [0.00003, 0.0001]:
                            for lr_immune in [0.00003, 0.0001]:
                                for extra_h in [True, False]:
                                    for extra_cdk in [True, False]:
                                        for extra_chemo in [True, False]:
                                            for extra_immune in [True, False]:
                                                for lr_loss in [0.001]:
                                                    try:
                                                    
                                                        dataset = TensorDataset(Tensor(train_x),
                                                                        Tensor(train_y))
                                                        
                                                        torch.manual_seed(1)
                                                        train_loader = DataLoader(dataset, batch_size = bs, shuffle = True)
                                                
                                                        val_dataset = TensorDataset(Tensor(val_x),
                                                                                Tensor(val_y))
                                                        torch.manual_seed(1)
                                                        val_loader = DataLoader(val_dataset, batch_size = val_x.shape[0], shuffle = True)
                                                
                                                
                                                        model = MultiTaskResponse(input_dim = train_x.shape[1], hidden_dim = hd, hormone_extra=extra_h, immune_extra=extra_immune, cdk_extra=extra_cdk, chemo_extra=extra_chemo)
                                                        
                                                        cindex = ConcordanceIndex()
                                                        custom_loss_function = MultiTaskLoss()
                                                        
                                                        optimizer = AdamW([
                                                            {"params": model.combined.parameters(), "lr": lr, 'weight_decay':0.0001},
                                                            {"params": model.pred_hormone.parameters(), "lr": lr_h, 'weight_decay':0},
                                                            {"params": model.pred_chemo.parameters(), "lr": lr_chemo, 'weight_decay':0},
                                                            {"params": model.pred_immune.parameters(), "lr": lr_immune, 'weight_decay':0},
                                                            {"params": model.pred_cdk.parameters(), "lr": lr_cdk, 'weight_decay':0},
                                                            {"params": custom_loss_function.parameters(), "lr": lr_loss, "weight_decay": 0},
                                                        ])
                                                        
                                                        
                                                        early_stopper = EarlyStopping(
                                                            patience=10,
                                                            min_delta=0,
                                                            restore_best_weights=True
                                                        )
                                                        
                                                        epochs = 1000
                                                
                                                
                                                        model.train()
                                                        
                                                        torch.manual_seed(1)
                                                        for epoch in range(epochs):
                                                            
                                                            train_loss = 0
                                                            for batch_idx, (x, y) in enumerate(train_loader):
                                                                x = x.to(DEVICE)
                                                                optimizer.zero_grad()
                                                        
                                                                hormone_out, chemo_out, immune_out, cdk_out = model(x)
                                                        
                                                                loss = custom_loss_function(hormone_out, chemo_out, immune_out, cdk_out, y)
                                                                
                                                                loss.backward()
                                                                optimizer.step()
                                                                train_loss += loss.item()
                                                        
                                                            train_loss /= len(train_loader)
                                                
                                                
                                                        # ---- Validation ----
                                                            model.eval()
                                                            val_loss = 0.0
                                                        
                                                            with torch.no_grad():
                                                                for batch_idx, (x, y) in enumerate(val_loader):
                                                                    x = x.to(DEVICE)
                                                                    
                                                                    hormone_out, chemo_out, immune_out, cdk_out = model(x)
                                                        
                                                                    loss = custom_loss_function(hormone_out, chemo_out, immune_out, cdk_out, y)
                                                        
                                                                    val_loss += loss.item()
                                                        
                                                            val_loss /= len(val_loader)
                                                            
                                                            df = torch.cat((hormone_out, y[:,1].bool().unsqueeze(1), y[:,0].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci1 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                        
                                                            df = torch.cat((chemo_out, y[:,7].bool().unsqueeze(1), y[:,6].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci2 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                        
                                                            df = torch.cat((immune_out, y[:,5].bool().unsqueeze(1), y[:,4].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci3 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                        
                                                            df = torch.cat((cdk_out, y[:,3].bool().unsqueeze(1), y[:,2].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci4 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                        
                                                            ci_miss = 1 - (ci1 + ci2 + ci3 + ci4)/4
                                                            print(
                                                                f"Epoch {epoch+1:03d} | "
                                                                f"Train Loss: {train_loss:.4f} | "
                                                                f"CI: {1-ci_miss}"
                                                            )
                                                        
                                                            # ---- Early stopping ----
                                                            early_stopper(ci_miss, model)
                                                            if early_stopper.early_stop:
                                                                print(f"Early stopping triggered at epoch {epoch+1}")
                                                                break
                                                
                                                
                                                
                                                        model.eval()
                                                        with torch.no_grad():
                                                            hormone_out, chemo_out, immune_out, cdk_out = model(train_x)
                                                            cindex = ConcordanceIndex()
                                                        
                                                            df = torch.cat((hormone_out, train_y[:,1].bool().unsqueeze(1), train_y[:,0].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci1 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            train_ci.append(ci1.item())
                                                            treat_list.append("Hormone Therapy")
                                                            bs_list.append(bs)
                                                            hd_list.append(hd)
                                                            lr_list.append(lr)
                                                            lr_cdk_list.append(lr_cdk)
                                                            lr_chemo_list.append(lr_chemo)
                                                            lr_hormone_list.append(lr_h)
                                                            lr_immune_list.append(lr_immune)
                                                            lr_loss_list.append(lr_loss)
                                                            extra_cdk_list.append(extra_cdk)
                                                            extra_chemo_list.append(extra_chemo)
                                                            extra_hormone_list.append(extra_h)
                                                            extra_immune_list.append(extra_immune)
                                                        
                                                            df = torch.cat((chemo_out, train_y[:,7].bool().unsqueeze(1), train_y[:,6].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci2 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            train_ci.append(ci2.item())
                                                            treat_list.append("Chemotherapy")
                                                            bs_list.append(bs)
                                                            hd_list.append(hd)
                                                            lr_list.append(lr)
                                                            lr_cdk_list.append(lr_cdk)
                                                            lr_chemo_list.append(lr_chemo)
                                                            lr_hormone_list.append(lr_h)
                                                            lr_immune_list.append(lr_immune)
                                                            lr_loss_list.append(lr_loss)
                                                            extra_cdk_list.append(extra_cdk)
                                                            extra_chemo_list.append(extra_chemo)
                                                            extra_hormone_list.append(extra_h)
                                                            extra_immune_list.append(extra_immune)
                                                        
                                                            df = torch.cat((immune_out, train_y[:,5].bool().unsqueeze(1), train_y[:,4].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci3 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            train_ci.append(ci3.item())
                                                            treat_list.append("Immunotherapy")
                                                            bs_list.append(bs)
                                                            hd_list.append(hd)
                                                            lr_list.append(lr)
                                                            lr_cdk_list.append(lr_cdk)
                                                            lr_chemo_list.append(lr_chemo)
                                                            lr_hormone_list.append(lr_h)
                                                            lr_immune_list.append(lr_immune)
                                                            lr_loss_list.append(lr_loss)
                                                            extra_cdk_list.append(extra_cdk)
                                                            extra_chemo_list.append(extra_chemo)
                                                            extra_hormone_list.append(extra_h)
                                                            extra_immune_list.append(extra_immune)
                                                
                                                            df = torch.cat((cdk_out, train_y[:,3].bool().unsqueeze(1), train_y[:,2].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci4 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            train_ci.append(ci4.item())
                                                            treat_list.append("CDK Inhibitor")
                                                            bs_list.append(bs)
                                                            hd_list.append(hd)
                                                            lr_list.append(lr)
                                                            lr_cdk_list.append(lr_cdk)
                                                            lr_chemo_list.append(lr_chemo)
                                                            lr_hormone_list.append(lr_h)
                                                            lr_immune_list.append(lr_immune)
                                                            lr_loss_list.append(lr_loss)
                                                            extra_cdk_list.append(extra_cdk)
                                                            extra_chemo_list.append(extra_chemo)
                                                            extra_hormone_list.append(extra_h)
                                                            extra_immune_list.append(extra_immune)
                                                
                                                
                                                ### VALIDATION
                                                        with torch.no_grad():
                                                            hormone_out, chemo_out, immune_out, cdk_out = model(val_x)
                                                            cindex = ConcordanceIndex()
                                                        
                                                            df = torch.cat((hormone_out, val_y[:,1].bool().unsqueeze(1), val_y[:,0].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci1 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            val_ci.append(ci1.item())
                                                        
                                                            df = torch.cat((chemo_out, val_y[:,7].bool().unsqueeze(1), val_y[:,6].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci2 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            val_ci.append(ci2.item())
                                                        
                                                            df = torch.cat((immune_out, val_y[:,5].bool().unsqueeze(1), val_y[:,4].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci3 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            val_ci.append(ci3.item())
                                                
                                                            df = torch.cat((cdk_out, val_y[:,3].bool().unsqueeze(1), val_y[:,2].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci4 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            val_ci.append(ci4.item())
                                                
                                                
                                                        with torch.no_grad():
                                                            hormone_out, chemo_out, immune_out, cdk_out = model(test_x)
                                                            cindex = ConcordanceIndex()
                                                        
                                                            df = torch.cat((hormone_out, test_y[:,1].bool().unsqueeze(1), test_y[:,0].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci1 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            test_ci.append(ci1.item())
                                                        
                                                            df = torch.cat((chemo_out, test_y[:,7].bool().unsqueeze(1), test_y[:,6].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci2 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            test_ci.append(ci2.item())
                                                        
                                                            df = torch.cat((immune_out, test_y[:,5].bool().unsqueeze(1), test_y[:,4].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci3 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            test_ci.append(ci3.item())
                                                
                                                            df = torch.cat((cdk_out, test_y[:,3].bool().unsqueeze(1), test_y[:,2].unsqueeze(1)), dim=1)
                                                            mask = ~torch.any(torch.isnan(df), dim=1)
                                                            df_filt = df[mask]
                                                            ci4 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
                                                            test_ci.append(ci4.item())
                                                    except:
                                                        pass
    
    
    dict = {'batch_size':bs_list, 'hidden_dimensions':hd_list, 
           'train_ci':train_ci, 'test_ci':test_ci, 'val_ci':val_ci,
            'shared_lr':lr_list, 'hormone_lr':lr_hormone_list, 'cdk_lr':lr_cdk_list,
            'chemo_lr':lr_chemo_list, 'immune_lr':lr_immune_list, 'loss_lr':lr_loss_list,
            'treatment':treat_list, 'extra_hormone':extra_hormone_list, 'extra_cdk':extra_cdk_list,
           'extra_chemo':extra_chemo_list, 'extra_immune':extra_immune_list}      
    df = pd.DataFrame(dict)
    
    print(df)
    
    df.to_csv(f"performance_fold{fold}.csv", index=False)
    

