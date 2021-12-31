from __future__ import division
import os
import zipfile
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import radians, cos, sin, asin, sqrt
# from sklearn.externals import joblib
import joblib
import scipy.io
import torch
from torch import nn
from scipy.stats import nbinom,norm
rand = np.random.RandomState(0)
"""
Geographical information calculation
"""
def get_long_lat(sensor_index,loc = None):
    """
        Input the index out from 0-206 to access the longitude and latitude of the nodes
    """
    if loc is None:
        locations = pd.read_csv('data/metr/graph_sensor_locations.csv')
    else:
        locations = loc
    lng = locations['longitude'].loc[sensor_index]
    lat = locations['latitude'].loc[sensor_index]
    return lng.to_numpy(),lat.to_numpy()

def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000


"""
Generate the training sample for forecasting task, same idea from STGCN
"""

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))


"""
Dynamically construct the adjacent matrix
"""

def get_Laplace(A):
    """
    Returns the laplacian adjacency matrix. This is for C_GCN
    """
    if A[0, 0] == 1:
        A = A - np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


def test_error_virtual(STmodel, unknow_set, test_data, A_s, E_maxvalue, Missing0):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """  
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension
    
    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs
   
    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index
    
    o = np.zeros([test_data.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]]) #Separate the test data into several h period
    
    for i in range(0, test_data.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        T_inputs = inputs*missing_inputs
        T_inputs = T_inputs/E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis = 0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        
        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    
    o = o*E_maxvalue 
    truth = test_inputs_s[0:test_data.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1]
    
    test_mask =  1 - missing_index_s[0:test_data.shape[0]//time_dim*time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    
    o_ = o[:,list(unknow_set)]
    truth_ = truth[:,list(unknow_set)]
    test_mask_ = test_mask[:,list(unknow_set)]

    MAE = np.sum(np.abs(o_ - truth_))/np.sum( test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_)*(o_ - truth_))/np.sum( test_mask_) )
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum( (o_ - truth_)*(o_ - truth_) )/np.sum( (truth_ - truth_.mean())*(truth_-truth_.mean() ) )
    print(truth_.mean())
    return MAE, RMSE, R2, o

def test_error(STmodel, unknow_set, test_data, A_s, E_maxvalue, Missing0):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """  
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension
    
    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs
   
    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index
    
    o = np.zeros([test_data.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]]) #Separate the test data into several h period
    
    for i in range(0, test_data.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        T_inputs = inputs*missing_inputs
        T_inputs = T_inputs/E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis = 0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        
        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    
    o = o*E_maxvalue 
    truth = test_inputs_s[0:test_data.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1]
    
    test_mask =  1 - missing_index_s[0:test_data.shape[0]//time_dim*time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    
    o_ = o[:,list(unknow_set)]
    truth_ = truth[:,list(unknow_set)]
    test_mask_ = test_mask[:,list(unknow_set)]

    MAE = np.sum(np.abs(o_ - truth_))/np.sum( test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_)*(o_ - truth_))/np.sum( test_mask_) )
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum( (o_ - truth_)*(o_ - truth_) )/np.sum( (truth_ - truth_.mean())*(truth_-truth_.mean() ) )
    print(truth_.mean())
    return MAE, RMSE, R2, o


def rolling_test_error(STmodel, unknow_set, test_data, A_s, E_maxvalue,Missing0):
    """
    :It only calculates the last time points' prediction error, and updates inputs each time point
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """  
    
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension
    
    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs
   
    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index

    o = np.zeros([test_data.shape[0] - time_dim, test_inputs_s.shape[1]])
    
    for i in range(0, test_data.shape[0] - time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        MF_inputs = inputs * missing_inputs
        MF_inputs = np.expand_dims(MF_inputs, axis = 0)
        MF_inputs = torch.from_numpy(MF_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        
        imputation = STmodel(MF_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i, :] = imputation[0, time_dim-1, :]
    
 
    truth = test_inputs_s[time_dim:test_data.shape[0]]
    o[missing_index_s[time_dim:test_data.shape[0]] == 1] = truth[missing_index_s[time_dim:test_data.shape[0]] == 1]
    
    o = o*E_maxvalue
    truth = test_inputs_s[0:test_data.shape[0]//time_dim*time_dim]
    test_mask =  1 - missing_index_s[time_dim:test_data.shape[0]]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
        
    MAE = np.sum(np.abs(o - truth))/np.sum( test_mask)
    RMSE = np.sqrt(np.sum((o - truth)*(o - truth))/np.sum( test_mask) )
    MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)  #avoid x/0
        
    return MAE, RMSE, MAPE, o

def test_error_cap(STmodel, unknow_set, full_set, test_set, A,time_dim,capacities):
    unknow_set = set(unknow_set)
    
    test_omask = np.ones(test_set.shape)
    test_omask[test_set == 0] = 0
    test_inputs = (test_set * test_omask).astype('float32')
    test_inputs_s = test_inputs#[:, list(proc_set)]

    
    missing_index = np.ones(np.shape(test_inputs))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index#[:, list(proc_set)]
    
    A_s = A#[:, list(proc_set)][list(proc_set), :]
    o = np.zeros([test_set.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]])
    
    for i in range(0, test_set.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        MF_inputs = inputs*missing_inputs
        MF_inputs = MF_inputs
        MF_inputs = np.expand_dims(MF_inputs, axis = 0)
        MF_inputs = torch.from_numpy(MF_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        
        imputation = STmodel(MF_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    
    o = o*capacities
    truth = test_inputs_s[0:test_set.shape[0]//time_dim*time_dim]
    truth = truth*capacities
    o[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1]
    o[truth == 0] = 0
    
    test_mask =  1 - missing_index_s[0:test_set.shape[0]//time_dim*time_dim]
    test_mask[truth == 0] = 0
    
    o_ = o[:,list(unknow_set)]
    truth_ = truth[:,list(unknow_set)]
    test_mask_ = test_mask[:,list(unknow_set)]

    MAE = np.sum(np.abs(o_ - truth_))/np.sum( test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_)*(o_ - truth_))/np.sum( test_mask_) )
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum( (o_ - truth_)*(o_ - truth_) )/np.sum( (truth_ - truth_.mean())*(truth_-truth_.mean() ) )
    print(truth_.mean())
    return MAE, RMSE, R2, o

def nb_nll_loss(y,n,p,y_mask=None):
    """
    y: true values
    y_mask: whether missing mask is given
    """
    nll = torch.lgamma(n) + torch.lgamma(y+1) - torch.lgamma(n+y) - n*torch.log(p) - y*torch.log(1-p)
    if y_mask is not None:
        nll = nll*y_mask
    return torch.sum(nll)

def nb_zeroinflated_nll_loss(y,n,p,pi,y_mask=None):
    """
    y: true values
    y_mask: whether missing mask is given
    https://stats.idre.ucla.edu/r/dae/zinb/
    """
    idx_yeq0 = y==0
    idx_yg0  = y>0
    
    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = y[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = y[idx_yg0]

    #L_yeq0 = torch.log(pi_yeq0) + (1-pi_yeq0)*torch.pow(p_yeq0,n_yeq0)
    #L_yg0  = torch.log(pi_yg0) + torch.lgamma(n_yg0+yg0) - torch.lgamma(yg0+1) - torch.lgamma(n_yg0) + n_yg0*torch.log(p_yg0) + yg0*torch.log(1-p_yg0)
    L_yeq0 = torch.log(pi_yeq0) + torch.log( (1-pi_yeq0)*torch.pow(p_yeq0,n_yeq0))
    L_yg0  = torch.log(1-pi_yg0) + torch.lgamma(n_yg0+yg0) - torch.lgamma(yg0+1) - torch.lgamma(n_yg0) + n_yg0*torch.log(p_yg0) + yg0*torch.log(1-p_yg0)
    #print('nll',torch.mean(L_yeq0),torch.mean(L_yg0),torch.mean(torch.log(pi_yeq0)),torch.mean(torch.log(pi_yg0)))
    return -torch.sum(L_yeq0)-torch.sum(L_yg0)

def nb_zeroinflated_draw(n,p,pi):
    """
    input: n, p, pi tensors
    output: drawn values
    """
    origin_shape = n.shape
    n = n.flatten()
    p = p.flatten()
    pi = pi.flatten()
    nb = nbinom(n,p)
    x_low = nb.ppf(0.01)
    x_up  = nb.ppf(0.99)
    pred = np.zeros_like(n)
   # print(n.shape,x_low.shape,pi.min())
    for i in range(len(x_low)):
        if x_up[i]<=1:
            x_up[i] = 1
        x = np.arange(x_low[i],x_up[i])
        #print(pi[0],pi[0].shape,x.shape,pi.shape)
        prob = (1-pi[i]) * nbinom.pmf(x,n[i],p[i])
#        print(len(prob),len(pi),len(n),len(x))
        prob[0] += pi[i] # zero-inflatted
        pred[i] = rand.choice(a=x,p=prob/np.sum(prob)) # random seed fixed, defined in the beginning

    return pred.reshape(origin_shape)

def gauss_draw(loc,scale):
    """
    input: n, p, pi tensors
    output: drawn values
    """
    origin_shape = loc.shape
    loc = loc.flatten()
    scale = scale.flatten()
    gauss = norm(loc,scale)
    x_low = gauss.ppf(0.01)
    x_up  = gauss.ppf(0.99)
    pred = np.zeros_like(loc)
    #print(n.shape,x_low.shape,pi.min())
    for i in range(len(x_low)):
        x = np.arange(x_low[i],x_up[i],100)
        prob = norm.pdf(x,loc[i],scale[i])
        pred[i] = rand.choice(a=x,p=prob/np.sum(prob)) # random seed fixed, defined in the beginning

    return pred.reshape(origin_shape)

def nb_draw(n,p):
    """
    input: n, p, pi tensors
    output: drawn values
    """
    origin_shape = n.shape
    n = n.flatten()
    p = p.flatten()
    nb = nbinom(n,p)
    x_low = nb.ppf(0.01)
    x_up  = nb.ppf(0.99)
    pred = np.zeros_like(n)
    for i in range(len(x_low)):
        if x_up[i]<=1:
            x_up[i] = 1
        if x_up[i] == x_low[i]:
            x_up[i] = x_low[i]+1
        #print(x_low[i],x_up[i])
        x = np.arange(x_low[i],x_up[i])
        prob = nbinom.pmf(x,n[i],p[i])
        pred[i] = rand.choice(a=x,p=prob/np.sum(prob)) # random seed fixed, defined in the beginning

    return pred.reshape(origin_shape)

def gauss_loss(y,loc,scale,y_mask=None):
    """
    The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.
    http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
    """
    torch.pi = torch.acos(torch.zeros(1)).item() * 2 # ugly define pi value in torch format
    LL = -1/2 * torch.log(2*torch.pi*torch.pow(scale,2)) - 1/2*( torch.pow(y-loc,2)/torch.pow(scale,2) )
    return -torch.sum(LL)
