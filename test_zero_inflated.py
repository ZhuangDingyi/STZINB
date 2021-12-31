from __future__ import division
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from utils import generate_dataset, get_normalized_adj, get_Laplace, calculate_random_walk_matrix,nb_zeroinflated_nll_loss,nb_zeroinflated_draw
from model import *
import random,os,copy
import math
import tqdm
from scipy.stats import nbinom
import pickle as pk
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
torch.manual_seed(0)
device = torch.device('cpu') 
num_timesteps_output = 4 
num_timesteps_input = num_timesteps_output

best_model = 'pth/STZINB_ny_full_5min.pth' #place it with other models
STmodel = torch.load(best_model).to(device=device)
# Load dataset
A = np.load('ny_data_full_5min/adj_rand0.npy')
X = np.load('ny_data_full_5min/cta_samp_rand0.npy')

space_dim = X.shape[1]
batch_size = 4
hidden_dim_s = 70
hidden_dim_t = 7
rank_s = 20
rank_t = 4
# Initial networks
TCN1 = B_TCN(space_dim, hidden_dim_t, kernel_size=3).to(device='cpu')
TCN2 = B_TCN(hidden_dim_t, rank_t, kernel_size = 3, activation = 'linear').to(device='cpu')
TCN3 = B_TCN(rank_t, hidden_dim_t, kernel_size= 3).to(device='cpu')
TNB = NBNorm_ZeroInflated(hidden_dim_t,space_dim).to(device='cpu')
SCN1 = D_GCN(num_timesteps_input, hidden_dim_s, 3).to(device='cpu')
SCN2 = D_GCN(hidden_dim_s, rank_s, 2, activation = 'linear').to(device='cpu')
SCN3 = D_GCN(rank_s, hidden_dim_s, 2).to(device='cpu')
SNB = NBNorm_ZeroInflated(hidden_dim_s,num_timesteps_output).to(device='cpu')
STmodel = ST_NB_ZeroInflated(SCN1, SCN2, SCN3, TCN1, TCN2, TCN3, SNB,TNB).to(device='cpu')

STmodel.load_state_dict(torch.load(best_model,map_location='cpu').state_dict())

X = X.T
X = X.astype(np.float32)
X = X.reshape((X.shape[0],1,X.shape[1]))
split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.7)
print(X.shape,A.shape)
max_value = np.max(X[:, :, :split_line1])

train_original_data = X[:, :, :split_line1]
val_original_data = X[:, :, split_line1:split_line2]
test_original_data = X[:, :, split_line2:]
training_input, training_target = generate_dataset(train_original_data,
                                                    num_timesteps_input=num_timesteps_input,
                                                    num_timesteps_output=num_timesteps_output)
val_input, val_target = generate_dataset(val_original_data,
                                            num_timesteps_input=num_timesteps_input,
                                            num_timesteps_output=num_timesteps_output)
test_input, test_target = generate_dataset(test_original_data,
                                            num_timesteps_input=num_timesteps_input,
                                            num_timesteps_output=num_timesteps_output)
print('input shape: ',training_input.shape,val_input.shape,test_input.shape)

A_wave = get_normalized_adj(A)
A_q = torch.from_numpy((calculate_random_walk_matrix(A_wave).T).astype('float32')).to(device=device)
A_h = torch.from_numpy((calculate_random_walk_matrix(A_wave.T).T).astype('float32')).to(device=device)
A_q = A_q.to(device=device)
A_h = A_h.to(device=device)
STmodel.eval()
with torch.no_grad():
    
    test_input = test_input.to(device='cpu')#.to(device=device)
    test_target = test_target.to(device='cpu')#.to(device=device)
    print(test_input.is_cuda,A_q.is_cuda,A_h.is_cuda)

    test_loss_all = []
    test_pred_all = np.zeros_like(test_target)
    n_test_all = np.zeros_like(test_target)
    p_test_all = np.zeros_like(test_target)
    pi_test_all = np.zeros_like(test_target)
    print(test_input.shape,test_target.shape)
    for i in range(0,test_input.shape[0],batch_size):
        x_batch = test_input[i:i+batch_size]
        n_test,p_test,pi_test = STmodel(x_batch,A_q,A_h)
        test_loss    = nb_zeroinflated_nll_loss(test_target[i:i+batch_size],n_test,p_test,pi_test).to(device="cpu")
        test_loss = np.asscalar(test_loss.detach().numpy())
        
        mean_pred =  (1-pi_test.detach().cpu().numpy())*(n_test.detach().cpu().numpy()/p_test.detach().cpu().numpy()-n_test.detach().cpu().numpy())
        test_pred_all[i:i+batch_size] = mean_pred
        n_test_all[i:i+batch_size] = n_test
        p_test_all[i:i+batch_size] = p_test
        pi_test_all[i:i+batch_size] = pi_test
        test_loss_all.append(test_loss)
    # The error of each horizon
    mae_list = []
    rmse_list=[]
    mape_list=[]
    for horizon in range(test_pred_all.shape[2]):
    
        mae  = np.mean(np.abs(test_pred_all[:,:,horizon] - test_target[:,:,horizon].detach().cpu().numpy()))
        rmse = np.sqrt(np.mean(test_pred_all[:,:,horizon] - test_target[:,:,horizon].detach().cpu().numpy()))
        mape = np.mean(np.abs( (test_pred_all[:,:,horizon] - test_target[:,:,horizon].detach().cpu().numpy())/(test_target[:,:,horizon].detach().cpu().numpy()+1e-5) ))

        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)
        print('Horizon %d MAE:%.4f RMSE:%.4f MAPE:%.4f'%(horizon,mae,rmse,mape))
    print('BestModel %s overall score: NLL %.5f; mae %.4f; rmse %.4f; mape %.4f'%(best_model,test_loss,np.mean(mae_list),np.mean(rmse_list),np.mean(mape_list)))
np.savez_compressed('output/ny_full_5min_ZISTNB',target=test_target.detach().cpu().numpy(),max_value=max_value,mean_pred=test_pred_all,n=n_test_all,p=p_test_all,pi=pi_test_all)
