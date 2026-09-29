import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from torchsurv.loss import cox
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###### Data Loading

treat_test = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//treatment_fold2_test.csv")
pathways_test = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//pathway_fold2_test.csv")
train_y = pd.read_csv(f"~//Documents//COH Breast Cohort//Final Datasets//survival_group_fold2_test.csv")

train_x = pd.concat([pathways_test, treat_test], axis=1)
train_x_pd = train_x

train_x = train_x.to_numpy()
train_y = train_y.to_numpy()

xmin = np.amin(train_x)
xmax = np.amax(train_x)
train_x = (train_x - xmin) / (xmax - xmin)

train_x = torch.from_numpy(train_x.astype(np.float32))
train_y = torch.from_numpy(train_y.astype(np.float32))


dataset = TensorDataset(Tensor(train_x),
                        Tensor(train_y))
torch.manual_seed(1)
train_loader = DataLoader(dataset, batch_size = 64, shuffle = True)

######## Model Definition

def Multitask_Total(input_dim, data,
                 DEVICE = DEVICE):
    

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

    torch.manual_seed(1)
    
    model = MultiTaskResponse(input_dim = train_x.shape[1], hidden_dim = 5000, hormone_extra=True,
            immune_extra=False,
            cdk_extra=True,
            chemo_extra=True)
    
    cindex = ConcordanceIndex()
    custom_loss_function = MultiTaskLoss()
    
    optimizer = AdamW([{"params": model.combined.parameters(), "lr": 0.00001, 'weight_decay':0.0001},
                       {"params": model.pred_hormone.parameters(), "lr": 0.00003, 'weight_decay':0},
                       {"params": model.pred_chemo.parameters(), "lr": 0.0001, 'weight_decay':0},
                       {"params": model.pred_immune.parameters(), "lr": 0.00003, 'weight_decay':0},
                       {"params": model.pred_cdk.parameters(), "lr": 0.00003, 'weight_decay':0},
                       {"params": custom_loss_function.parameters(), "lr": 0.001, "weight_decay": 0},
                                            ])
    
    
    train_loss_list = []
    val_loss_list = []
    val_ci_list = []
    
    early_stopper = EarlyStopping(
    patience=25,
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
        
        with torch.no_grad():
          
            hormone_out, chemo_out, immune_out, cdk_out = model(train_x)
            cindex = ConcordanceIndex()
        
            df = torch.cat((hormone_out, train_y[:,1].bool().unsqueeze(1), train_y[:,0].unsqueeze(1)), dim=1)
            mask = ~torch.any(torch.isnan(df), dim=1)
            df_filt = df[mask]
            ci1 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
        
            df = torch.cat((chemo_out, train_y[:,7].bool().unsqueeze(1), train_y[:,6].unsqueeze(1)), dim=1)
            mask = ~torch.any(torch.isnan(df), dim=1)
            df_filt = df[mask]
            ci2 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
        
            df = torch.cat((immune_out, train_y[:,5].bool().unsqueeze(1), train_y[:,4].unsqueeze(1)), dim=1)
            mask = ~torch.any(torch.isnan(df), dim=1)
            df_filt = df[mask]
            ci3 = cindex(df_filt[:,0], df_filt[:,1].bool(), df_filt[:,2])
        
            df = torch.cat((cdk_out, train_y[:,3].bool().unsqueeze(1), train_y[:,2].unsqueeze(1)), dim=1)
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

    return model


torch.manual_seed(1)
mod = Multitask_Total(input_dim = train_x.shape[1], data = dataset,
                   DEVICE = DEVICE)


torch.save(mod.state_dict(), 'multitask_model_finalweights.pth')