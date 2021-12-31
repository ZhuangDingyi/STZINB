import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys
import math
from scipy.stats import nbinom
from torch.nn.utils import weight_norm

# Define the NB class first, not mixture version
class NBNorm(nn.Module):
    def __init__(self, c_in, c_out):
        super(NBNorm,self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.p_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1,1),
                                    bias=True)
        self.out_dim = c_out # output horizon

    def forward(self,x):
        x = x.permute(0,2,1,3)
        (B, _, N,_) = x.shape # B: batch_size; N: input nodes
        n = self.n_conv(x).squeeze_(-1)
        p = self.p_conv(x).squeeze_(-1)

        # Reshape
        n = n.view([B,self.out_dim,N])
        p = p.view([B,self.out_dim,N])

        # Ensure n is positive and p between 0 and 1
        n = F.softplus(n) # Some parameters can be tuned here
        p = F.sigmoid(p)
        return n.permute([0,2,1]), p.permute([0,2,1])

    def likelihood_loss(self,y,n,p,y_mask=None):
        """
        y: true values
        y_mask: whether missing mask is given
        """
        nll = torch.lgamma(n) + torch.lgamma(y+1) - torch.lgamma(n+y) - n*torch.log(p) - y*torch.log(1-p)
        if y_mask is not None:
            nll = nll*y_mask
        return torch.sum(nll)

    def mean(self,n,p):
        """
        :param cat: Input data of shape (batch_size, num_timesteps, in_nodes)
        :return: Output data of shape (batch_size, 1, num_timesteps, in_nodes)
        """ 
        pass

# Define the Gaussian 
class GaussNorm(nn.Module):
    def __init__(self, c_in, c_out):
        super(GaussNorm,self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.p_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1,1),
                                    bias=True)
        self.out_dim = c_out # output horizon

    def forward(self,x):
        x = x.permute(0,2,1,3)
        (B, _, N,_) = x.shape # B: batch_size; N: input nodes
        loc    = self.n_conv(x).squeeze_(-1) # The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.
        scale  = self.p_conv(x).squeeze_(-1)

        # Reshape
        loc   = loc.view([B,self.out_dim,N])
        scale = scale.view([B,self.out_dim,N])

        # Ensure n is positive and p between 0 and 1
        loc = F.softplus(loc) # Some parameters can be tuned here, count data are always positive
        scale = F.sigmoid(scale)
        return loc.permute([0,2,1]), scale.permute([0,2,1])

# Define the NB class first, not mixture version
class NBNorm_ZeroInflated(nn.Module):
    def __init__(self, c_in, c_out):
        super(NBNorm_ZeroInflated,self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.p_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1,1),
                                    bias=True)

        self.pi_conv = nn.Conv2d(in_channels=c_in,
                                    out_channels=c_out,
                                    kernel_size=(1,1),
                                    bias=True)
                                    
        self.out_dim = c_out # output horizon

    def forward(self,x):
        x = x.permute(0,2,1,3)
        (B, _, N,_) = x.shape # B: batch_size; N: input nodes
        n  = self.n_conv(x).squeeze_(-1)
        p  = self.p_conv(x).squeeze_(-1)
        pi = self.pi_conv(x).squeeze_(-1)

        # Reshape
        n = n.view([B,self.out_dim,N])
        p = p.view([B,self.out_dim,N])
        pi = pi.view([B,self.out_dim,N])

        # Ensure n is positive and p between 0 and 1
        n = F.softplus(n) # Some parameters can be tuned here
        p = F.sigmoid(p)
        pi = F.sigmoid(pi)
        return n.permute([0,2,1]), p.permute([0,2,1]), pi.permute([0,2,1])

class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """       
    def __init__(self, in_channels, out_channels, orders, activation = 'relu'): 
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                             out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)
        
    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)
        
    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0] # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length
        supports = []
        supports.append(A_q)
        supports.append(A_h)
        
        x0 = X.permute(1, 2, 0) #(num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[num_node, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1
                
        x = torch.reshape(x, shape=[self.num_matrices, num_node, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])         
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)     
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)   
            
        return x


## Code of BTCN from Yuankai
class B_TCN(nn.Module):
    """
    Neural network block that applies a bidirectional temporal convolution to each node of
    a graph.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,activation = 'relu',device='cpu'):
        """
        :param in_channels: Number of nodes in the graph.
        :param out_channels: Desired number of output features.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(B_TCN, self).__init__()
        # forward dirction temporal convolution
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.activation = activation
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        
        self.conv1b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3b = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        
    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :return: Output data of shape (batch_size, num_timesteps, num_features)
        """
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        Xf = X.unsqueeze(1)  # (batch_size, 1, num_timesteps, num_nodes)
        
        inv_idx = torch.arange(Xf.size(2)-1, -1, -1).long().to(device=self.device)#.to(device=self.device).to(device=self.device)
        Xb = Xf.index_select(2, inv_idx) # inverse the direction of time
        
        Xf = Xf.permute(0, 3, 1, 2)
        Xb = Xb.permute(0, 3, 1, 2) #(batch_size, num_nodes, 1, num_timesteps)
        tempf = self.conv1(Xf) * torch.sigmoid(self.conv2(Xf)) #+
        outf = tempf + self.conv3(Xf) 
        outf = outf.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels])        
        
        tempb = self.conv1b(Xb) * torch.sigmoid(self.conv2b(Xb)) #+
        outb = tempb + self.conv3b(Xb)
        outb = outb.reshape([batch_size, seq_len - self.kernel_size + 1, self.out_channels])
        
        rec = torch.zeros([batch_size, self.kernel_size - 1, self.out_channels]).to(device=self.device)#.to(device=self.device)
        outf = torch.cat((outf, rec), dim = 1)
        outb = torch.cat((outb, rec), dim = 1) #(batch_size, num_timesteps, out_features)
        
        inv_idx = torch.arange(outb.size(1)-1, -1, -1).long().to(device=self.device)#.to(device=self.device)
        outb = outb.index_select(1, inv_idx)
        out = outf + outb
        if self.activation == 'relu':
            out = F.relu(outf) + F.relu(outb)
        elif self.activation == 'sigmoid':
            out = F.sigmoid(outf) + F.sigmoid(outb)       
        return out


class ST_NB(nn.Module):
    """
  wx_t  + wx_s
    |       |
   TC4     SC4
    |       |
   TC3     SC3
    |       |
   z_t     z_s
    |       |
   TC2     SC2
    |       |  
   TC1     SC1
    |       |
   x_m     x_m
    """
    def __init__(self, SC1, SC2, SC3, TC1, TC2, TC3, SNB,TNB): 
        super(ST_NB, self).__init__()
        self.TC1 = TC1
        self.TC2 = TC2
        self.TC3 = TC3
        self.TNB = TNB

        self.SC1 = SC1
        self.SC2 = SC2
        self.SC3 = SC3
        self.SNB = SNB

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """  
        X = X[:,:,:,0] # Dummy dimension deleted
        X_T = X.permute(0,2,1)
        X_t1 = self.TC1(X_T)
        X_t2 = self.TC2(X_t1) #num_time, rank
        self.temporal_factors = X_t2
        X_t3 = self.TC3(X_t2)
        _b,_h,_ht = X_t3.shape
        n_t_nb,p_t_nb = self.TNB(X_t3.view(_b,_h,_ht,1))

        X_s1 = self.SC1(X, A_q, A_h)
        X_s2 = self.SC2(X_s1, A_q, A_h) #num_nodes, rank
        self.space_factors = X_s2
        X_s3 = self.SC3(X_s2, A_q, A_h)
        _b,_n,_hs = X_s3.shape
        n_s_nb,p_s_nb = self.SNB(X_s3.view(_b,_n,_hs,1))
        n_res = n_t_nb.permute(0, 2, 1) * n_s_nb
        p_res = p_t_nb.permute(0, 2, 1) * p_s_nb
               
        return n_res,p_res

class ST_Gau(nn.Module):
    """
  wx_t  + wx_s
    |       |
   TC4     SC4
    |       |
   TC3     SC3
    |       |
   z_t     z_s
    |       |
   TC2     SC2
    |       |  
   TC1     SC1
    |       |
   x_m     x_m
    """
    def __init__(self, SC1, SC2, SC3, TC1, TC2, TC3, SGau,TGau): 
        super(ST_Gau, self).__init__()
        self.TC1 = TC1
        self.TC2 = TC2
        self.TC3 = TC3
        self.TGau = TGau

        self.SC1 = SC1
        self.SC2 = SC2
        self.SC3 = SC3
        self.SGau = SGau

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """  
        X = X[:,:,:,0] #.to(device='cuda') # Dummy dimension deleted
        X_T = X.permute(0,2,1)
        X_t1 = self.TC1(X_T)
        X_t2 = self.TC2(X_t1) #num_time, rank
        self.temporal_factors = X_t2
        X_t3 = self.TC3(X_t2)
        _b,_h,_ht = X_t3.shape
        loc_t,scale_t = self.TGau(X_t3.view(_b,_h,_ht,1))

        X_s1 = self.SC1(X, A_q, A_h)
        X_s2 = self.SC2(X_s1, A_q, A_h) #num_nodes, rank
        self.space_factors = X_s2
        X_s3 = self.SC3(X_s2, A_q, A_h)
        _b,_n,_hs = X_s3.shape
        loc_s,scale_s = self.SGau(X_s3.view(_b,_n,_hs,1))

        loc_res = loc_t.permute(0, 2, 1) * loc_s
        scale_res = scale_t.permute(0, 2, 1) * scale_s
               
        return loc_res,scale_res

class ST_NB_ZeroInflated(nn.Module):
    """
  wx_t  + wx_s
    |       |
   TC4     SC4
    |       |
   TC3     SC3
    |       |
   z_t     z_s
    |       |
   TC2     SC2
    |       |  
   TC1     SC1
    |       |
   x_m     x_m
    """
    def __init__(self, SC1, SC2, SC3, TC1, TC2, TC3, SNB,TNB): 
        super(ST_NB_ZeroInflated, self).__init__()
        self.TC1 = TC1
        self.TC2 = TC2
        self.TC3 = TC3
        self.TNB = TNB

        self.SC1 = SC1
        self.SC2 = SC2
        self.SC3 = SC3
        self.SNB = SNB

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_hat: The Laplacian matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """  
        X = X[:,:,:,0]#.to(device='cuda') # Dummy dimension deleted
        X_T = X.permute(0,2,1)
        X_t1 = self.TC1(X_T)
        X_t2 = self.TC2(X_t1) #num_time, rank
        self.temporal_factors = X_t2
        X_t3 = self.TC3(X_t2)
        _b,_h,_ht = X_t3.shape
        n_t_nb,p_t_nb,pi_t_nb = self.TNB(X_t3.view(_b,_h,_ht,1))

        X_s1 = self.SC1(X, A_q, A_h)
        X_s2 = self.SC2(X_s1, A_q, A_h) #num_nodes, rank
        self.space_factors = X_s2
        X_s3 = self.SC3(X_s2, A_q, A_h)
        _b,_n,_hs = X_s3.shape
        n_s_nb,p_s_nb,pi_s_nb = self.SNB(X_s3.view(_b,_n,_hs,1))
        n_res = n_t_nb.permute(0, 2, 1) * n_s_nb
        p_res = p_t_nb.permute(0, 2, 1) * p_s_nb
        pi_res = pi_t_nb.permute(0, 2, 1) * pi_s_nb

        return n_res,p_res,pi_res

