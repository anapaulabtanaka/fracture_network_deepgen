#!/usr/bin/env python
# coding: utf-8

# # Model for graph generation (topology) - training

# In[ ]:


import networkx
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# import pyvista as pv
import os

import pickle


# # In[ ]:
# 
# 
# # Choose backend for matplotlib
# # -----------------------------
# from IPython import get_ipython
# # get_ipython().run_line_magic('matplotlib', 'widget')
# get_ipython().run_line_magic('matplotlib', 'inline')
# 
# # Or simply:
# # %matplotlib widget
# # %matplotlib inline
# 
# 
# # In[ ]:
# 
# 
# # # Choose backend for pyvista with jupyter
# # # ---------------------------------------
# # # pv.set_jupyter_backend('trame')  # 3D-interactive plots
# # pv.set_jupyter_backend('static') # static plots
# 
# # # Notes:
# # # -> ignored if run in a standard python shell
# # # -> use keyword argument "notebook=False" in Plotter() to open figure in a pop-up window
# 
# 
# # ## Load local functions 
# 
# In[ ]:


print('Load local functions...')

# import sys
# sys.path.insert(1, '../utils/')

# from graph_utils import *
# from graph_rnn import *
# from ml_utils import *
 
with open('../utils/graph_utils.py') as f: exec(f.read())
with open('../utils/graph_rnn.py') as f: exec(f.read())
with open('../utils/ml_utils.py') as f: exec(f.read())


# ## Load parameters
# 
# Some parameters (dimension / attribute considered and indexes / parameters for plotting graphs)
# 

# In[ ]:


print('Load parameters...')

# from params import *

with open('params.py') as f: exec(f.read())


# ## Output settings
# For saving data set and model (once trained).

# In[ ]:


print('Define output settings...')

# Output directory (for saving)
# -----------------------------
out_dir = 'out_graphRNN_model' # output directory

fig_dir = 'fig'      # PARAMS

plt_show = False     # PARAMS (show graphics 2D ?)
# off_screen = True    # PARAMS (show graphics 3D ?)

save_fig_png = True  # PARAMS
fig_prefix = '01'    # PARAMS

fig_counter = 0

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)
    
# Files for saving data set (pickle) (see further)
# ------------------------------------------------
filename_data_set = os.path.join(out_dir, f'data_set.pickle')

# Files for saving network (rnn_G and rnn_E) (see further)
# --------------------------------------------------------
filename_hyper_param_G = os.path.join(out_dir, 'rnn_G_hyper_params.txt')
filename_hyper_param_E = os.path.join(out_dir, 'rnn_E_hyper_params.txt')

filename_param_G = os.path.join(out_dir, 'rnn_G.params')
filename_param_E = os.path.join(out_dir, 'rnn_E.params')

# Files for saving loss and lr (see further)
# -------------------------------------------
filename_loss_lr = os.path.join(out_dir, 'rnn_loss_lr.pickle')


# ## Data set

# ### Read graph collection - training set

# In[ ]:


print('Read data set (collection of subgraphs)...')

# Read from text files (only position attribute)
# ----
# Files
data_dir = 'data_gen'
filename_base = 'graph_collection_data_set'

# Load graph list
G_list = load_networkx_graph_list(
    data_dir, filename_base, 
    suffix_nodes='_nodes.dat', 
    suffix_edges='_links.dat', 
    delimiter_nodes=' ',  
    delimiter_edges=' ',
    node_attrs=['pos'],
    node_attrs_ind=[tuple(range(dim))],
    nodet_attrs_type=['float'],
    start_id_at_0=True)


# ### Show first graphs

# In[ ]:


print('Plot data set (collection of subgraphs) (topology)...')

# Plot first graphs - 2d - topology only
# ======================================
kwds = kwds_multi.copy()

figsize = figsize_multi
# -----

ng = 16

ng = min(len(G_list), ng)
nr = int(np.sqrt(ng))
nc = ng//nr + (ng%nr>0)

# Plot
# ----
plt.subplots(nr, nc, figsize=figsize)
for i, G in enumerate(G_list[:ng]):
    plt.subplot(nr, nc, i+1)
    networkx.draw(G, with_labels=False, **kwds)
    plt.title(f'n_nodes={G.number_of_nodes()}')

for i in range(ng, nr*nc):
    plt.subplot(nr, nc, i+1)
    plt.axis('off')

plt.suptitle(f'graphRNN - train set')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_graphRNN_train_set.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# In[ ]:


print(f'Number of graphs in list: {len(G_list)}')


# ### Define the data set

# In[ ]:


print('Define the data set for graphRNN...')

# Data set
# --------
G_nsample = np.full((len(G_list), ), 1) # number of times each graph in the list is sampled
                                          # sum(G_nsample) gives the size of the data set
# Parameters for encoding adjacency matrix
use_bfs = True
max_n_nodes = None 
max_prev_node = None
calc_max_prev_node_kwargs={'nsample':10000, 'quantile':1.0, 'seed':134}

data_set = Graph_sequence_sampler_data_set(
    G_list, G_nsample, use_bfs=use_bfs,
    max_n_nodes=max_n_nodes, max_prev_node=max_prev_node,
    calc_max_prev_node_kwargs=calc_max_prev_node_kwargs)

print(f'Data set:\n\
   size = {len(data_set)}\n\
   max_n_nodes   = {data_set.max_n_nodes:5d}\n\
   max_prev_node = {data_set.max_prev_node:5d}')


# In[ ]:


print('Get array of number of nodes in data set...')

# Get array of number of nodes in data set (via data loader, see below)
data_set_n_nodes = []
batch_size = 500
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)
for _, n_nodes in data_loader:
    data_set_n_nodes = data_set_n_nodes + n_nodes.tolist()

data_set_n_nodes = np.asarray(data_set_n_nodes)

# Get mean, min, max number of nodes in data set
data_set_n_nodes_mean = data_set_n_nodes.mean()
data_set_n_nodes_std = data_set_n_nodes.std()
data_set_n_nodes_min  = data_set_n_nodes.min()
data_set_n_nodes_max  = data_set_n_nodes.max()

print(f'Data set - number of nodes - mean: {data_set_n_nodes_mean:9.3f}')
print(f'Data set - number of nodes - std : {data_set_n_nodes_std:9.3f}')
print(f'Data set - number of nodes - min : {data_set_n_nodes_min:9.3f}')
print(f'Data set - number of nodes - max : {data_set_n_nodes_max:9.3f}')


# ### Load data via a data loader, and plot first encoded adjacency matrices

# In[ ]:


print('Define data loader...')

# Data loader (pytorch)
# ---------------------
batch_size = 6
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)


# In[ ]:


print('Plot first batches (topology)...')

torch.random.manual_seed(293) # -> for reproducibility of batches delivered by the data loader (if needed)

figsize = figsize_lh4

# Figure
for i, (x, n_nodes) in enumerate(data_loader):
    if i == 3:
        break
    plt.subplots(1, batch_size, figsize=figsize)
    #plt.clf() # clear figure
    plt.suptitle(f'Encoding adj. matrix (max_prev_node={data_set.max_prev_node})')
    for j in range(len(x)):
        plt.subplot(1, batch_size, j+1)
        m = x[j, :n_nodes[j]-1, :] # encoded adj. matrix
        plt.imshow(m, origin='upper', extent=[0.5, m.shape[1]+0.5, m.shape[0]+0.5, 0.5], interpolation='none')
        plt.gca().set_aspect(.5)
        plt.title(f'Batch #{i} : {j}')
    for j in range(len(x), batch_size):
        plt.subplot(1, batch_size, j+1)
        plt.axis('off')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_graphRNN_train_set_enc_ad_mat_batch_{i}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1


# ## RNN model for graph generation

# ### Define the model (design)

# In[ ]:


print('Define the model (design)...')

# RNN model for graph generation
# ------------------------------
# Two RNN models imbricated: rnn_G, rnn_E

# class RNN_model: see graph_rnn.py

# rnn_G (RNN model at graph level)
# ================================
# Hyper parameters (design of the model)
rnn_G_hyper_params = dict(
    input_size        = data_set.max_prev_node,   # FIXED (must not be changed!)
    embed_input       = True,
    embed_input_size  = 64,    # used if rnn_G_embed_input=True

    hidden_size       = 48,

    has_output        = True,   
    embed_output      = True,  # used if rnn_G_has_output=True
    embed_output_size = 16,    # used if rnn_G_has_output=True and rnn_G_embed_output=True
    output_size       = 32,    # used if rnn_G_has_output=True
                                    # note: if rnn_G_has_output=False, then 
                                    # rnn_G.output_size is set to rnn_G_hidden_size

    num_layers        = 4,
    rnn_type          = 'GRU', # {'RNN', 'GRU', 'LSTM'}
    dropout           = 0.0
)

# RNN model
rnn_G = RNN_model(**rnn_G_hyper_params)

# rnn_E (RNN model at edge level)
# ===============================
# Hyper parameters (design of the model)
rnn_E_hyper_params = dict(
    input_size        = 1,     # FIXED (must not be changed!)
    embed_input       = True,
    embed_input_size  = 24,    # used if rnn_E_embed_input=True

    hidden_size       = rnn_G.output_size,   # FIXED (must not be changed!)

    has_output        = True,   
    embed_output      = True,  # used if rnn_E_has_output=True
    embed_output_size = 36,    # used if rnn_E_has_output=True and rnn_E_embed_output=True
    output_size       = 1,     # used if rnn_E_has_output=True / FIXED (must not be changed!)
                                    # note: if rnn_E_has_output=False, then 
                                    # rnn_E.output_size is set to rnn_E_hidden_size
                                    # which should be 1 in this case

    num_layers        = 4,
    rnn_type          = 'GRU', # {'RNN', 'GRU', 'LSTM'}
    dropout           = 0.0
)

# RNN model
rnn_E = RNN_model(**rnn_E_hyper_params)


# ### Load the model (hyper parameters and parameters) - if existing and already trained

# In[ ]:


# # print('Load the model (hyper parameters and parameters) (graphRNN)...')

# # Load model

# # rnn_G 
# # =====
# # Hyper parameters (design of the model)
# with open(filename_hyper_param_G, 'r') as f: rnn_G_hyper_params = eval(f.read())

# # RNN model (parameters)
# rnn_G = RNN_model(**rnn_G_hyper_params)
# rnn_G.load_state_dict(torch.load(filename_param_G))

# # rnn_E 
# # =====
# # Hyper parameters (design of the model)
# with open(filename_hyper_param_E, 'r') as f: rnn_E_hyper_params = eval(f.read())

# # RNN model (parameters)
# rnn_E = RNN_model(**rnn_E_hyper_params)
# rnn_E.load_state_dict(torch.load(filename_param_E))


# In[ ]:


# # print('Load loss and lr...')

# # Load loss and lr
# with open(filename_loss_lr, 'rb') as f: loss, lr_used_G, lr_used_E = pickle.load(file=f)


# ### Display the model design

# In[ ]:


print('Display the model (graphRNN)...')

print('\n')
print('rnn_G\n-----')
print(rnn_G)
print(f'Number of (learnable) params: {nb_net_params(rnn_G)}')

print('\n')
print('rnn_E\n-----')
print(rnn_E)
print(f'Number of (learnable) params: {nb_net_params(rnn_E)}')


# ### Display the model parameters

# In[17]:


# rnn_G.state_dict() # display parameters
# rnn_E.state_dict() # display parameters


# ### Re-initialize the model parameters (if needed)

# In[ ]:


print('[Re-]initialize the model parameters...')

# Re-initialize the model parameters (if needed)
# -----------------------_----------------------
# Re-initialize rnn_G parameters
rnn_G_seed = 132
torch.random.manual_seed(rnn_G_seed)
# reset_all_parameters(rnn_G)
rnn_G.init_weights()

# Re-initialize rnn_E parameters
rnn_E_seed = 985
torch.random.manual_seed(rnn_E_seed)
# reset_all_parameters(rnn_E)
rnn_E.init_weights()

# print('rnn_G parameters\n-------------')
# for p in rnn_G.parameters():
#     print(f'- shape:', p.data.shape)
#     # print(f'- values:', p.data)

# print('rnn_E parameters\n--------------------')
# for p in rnn_E.parameters():
#     print(f'- shape:', p.data.shape)
#     # print(f'- values:', p.data)

# Initialize lists for sotring loss, lr
# -------------------------------------
loss = []
lr_used_G, lr_used_E = [], []


# ### Train the model

# In[ ]:


print('Train the model...')

# Re-launch as many times as needed (and change the settings below if needed)!
# ----------------------------------------------------------------------------

# Settings
# ========

# Batch size, number of epochs
# ----------------------------
batch_size = 50
num_epochs = int(5.0*(data_set_n_nodes_mean+3*data_set_n_nodes_std))
print_epoch = 1

print(f'... num_epochs = {num_epochs}')

# Optimizer for rnn_G
# -------------------
# # - 1. Stochastic Gradient Descent (SGD)
# lr_G = 0.3             # learning rate
# weight_decay_G = 0.001 # L2-regularization
# momentum_G = 0.0       # momentum
# optimizer_G = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
# - 2. Adam
lr_G = 0.0025           # learning rate; default: lr=0.001
betas_G = (0.9, 0.999)   # betas parameters; default: betas=(0.9, 0.999)
eps_G = 1.e-8          # epsilon parameter; default: 1.e-8
weight_decay_G = 0.0   # L2-regularization; default: 0.0
optimizer_G = torch.optim.Adam(rnn_G.parameters(), lr=lr_G, betas=betas_G, eps=eps_G, weight_decay=weight_decay_G)
# ...

# Learning rate scheduler for rnn_G
# ---------------------------------
# # - 1. CosineAnnealingLR
# lr_init_G = lr_G
# T_max_G = num_epochs
# eta_min_G = lr_init_G / 10.
# lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=T_max_G, eta_min=eta_min_G, last_epoch=-1, verbose=False)
# # Essentially, with eta_max = lr_init, at epoch t, the learning rate is set to
# #    eta_t = eta_min + 1/2 * (eta_max - eta_min) * (1 + cos(t/T_max*pi))
#
# - 2. CosineAnnealingWarmRestarts
lr_init_G = lr_G
eta_min_G = lr_init_G / 10.
T_mult_G = 2
nrestart_G = 3
T_0_G = int(np.ceil(num_epochs * (T_mult_G-1) / (T_mult_G**(nrestart_G+1)-1)))
lr_scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, T_0=T_0_G, T_mult=T_mult_G, eta_min=eta_min_G, last_epoch=-1, verbose=False)
# Essentially, with eta_max = lr_init, at epoch t, the learning rate is set to
#    eta_t = eta_min + 1/2 * (eta_max - eta_min) * (1 + cos(T_cur/T_i*pi))
# where
#    T_cur: the number of epochs since the last restart
#    T_i  : the number of epochs between two warm restarts
#    T_0  : number of epochs before the 1st warm restart
#    T_mult: T_i is defined as T_i = T_mult * T_{i-1}
#
# # - 3. MultiStepLR
# gamma_G = .3
# milestones_G = [int(num_epochs/9), int(num_epochs/3)]
# lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, milestones_G, gamma=gamma_G, last_epoch=-1, verbose=False)
# # At epoch in `milestones`, lr is multiplied by `gamma`
# # - 4
# lr_scheduler_G = None
# ...

# Optimizer for rnn_E
# -------------------
# # - 1. Stochastic Gradient Descent (SGD)
# lr_E = 0.3             # learning rate
# weight_decay_E = 0.001 # L2-regularization
# momentum_E = 0.0       # momentum
# optimizer_E = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
# - 2. Adam
lr_E = 0.0025           # learning rate; default: lr=0.001
betas_E = (0.9, 0.999)   # betas parameters; default: betas=(0.9, 0.999)
eps_E = 1.e-8          # epsilon parameter; default: 1.e-8
weight_decay_E = 0.0   # L2-regularization; default: 0.0
optimizer_E = torch.optim.Adam(rnn_E.parameters(), lr=lr_E, betas=betas_E, eps=eps_E, weight_decay=weight_decay_E)
# ...

# Learning rate scheduler for rnn_E
# ---------------------------------
# # - 1. CosineAnnealingLR
# lr_init_E = lr_E
# T_max_E = num_epochs
# eta_min_E = lr_init_E / 10.
# lr_scheduler_E = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_E, T_max=T_max_E, eta_min=eta_min_E, last_epoch=-1, verbose=False)
# # Essentially, with eta_max = lr_init, at epoch t, the learning rate is set to
# #    eta_t = eta_min + 1/2 * (eta_max - eta_min) * (1 + cos(t/T_max*pi))
#
# - 2. CosineAnnealingWarmRestarts
lr_init_E = lr_E
eta_min_E = lr_init_E / 10.
T_mult_E = 2
nrestart_E = 3
T_0_E = int(np.ceil(num_epochs * (T_mult_E-1) / (T_mult_E**(nrestart_E+1)-1)))
lr_scheduler_E = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_E, T_0=T_0_E, T_mult=T_mult_E, eta_min=eta_min_E, last_epoch=-1, verbose=False)
# Essentially, with eta_max = lr_init, at epoch t, the learning rate is set to
#    eta_t = eta_min + 1/2 * (eta_max - eta_min) * (1 + cos(T_cur/T_i*pi))
# where
#    T_cur: the number of epochs since the last restart
#    T_i  : the number of epochs between two warm restarts
#    T_0  : number of epochs before the 1st warm restart
#    T_mult: T_i is defined as T_i = T_mult * T_{i-1}
#
# # - 3. MultiStepLR
# gamma_E = .3
# milestones_E = [int(num_epochs/9), int(num_epochs/3)]
# lr_scheduler_E = torch.optim.lr_scheduler.MultiStepLR(optimizer_E, milestones_E, gamma=gamma_E, last_epoch=-1, verbose=False)
# # At epoch in `milestones`, lr is multiplied by `gamma`
# # - 4
# lr_scheduler_E = None
# ...

# ----
# # Note: to compute the sequence of lr used (for rnn_G), and plot it: 
# lr_used_G = []
# for i in range(num_epochs):
#     lr_used_G.append(lr_scheduler_G.get_last_lr()[0])
#     optimizer_G.step()
#     lr_scheduler_G.step()
# plt.grid()
# plt.show()
# ========

# Create Data Loader
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

# Train
t1 = time.time()
loss_cur, lr_used_G_cur, lr_used_E_cur = \
    train_rnn_model_graph_gen(
        rnn_G, 
        rnn_E, 
        data_loader,
        optimizer_G,
        optimizer_E,
        lr_scheduler_G=lr_scheduler_G,
        lr_scheduler_E=lr_scheduler_E,
        return_lr=True,
        return_loss=True,
        num_epochs=num_epochs,
        print_epoch=print_epoch,
        # device=torch.device('cpu')
        device=torch.device('cuda:0')
    )
t2 = time.time()

# Update lists of loss, lr
loss = loss + loss_cur # concatenate list

lr_used_G = lr_used_G + lr_used_G_cur # concatenate list
lr_used_E = lr_used_E + lr_used_E_cur # concatenate list

# Print elapsed time and result of last epoch
print(f'Elapsed time for {num_epochs} epochs: {t2-t1:.3g} s')
print(f'Last epoch, loss = {loss_cur[-1]:.5g}')


# In[ ]:


print('Plot loss...')

# Plot loss
# ---------
color_train = 'tab:blue'

figsize = figsize_lh3

plt.figure(figsize=figsize)
plt.plot(loss, ls='-', color=color_train, label='loss')
#plt.yscale('log')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.title('loss')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_graphRNN_loss.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# In[ ]:


print('Plot loss (log scale)...')

# Plot loss (log scale along y axis)
# ---------
color_train = 'tab:blue'

figsize = figsize_lh3

plt.figure(figsize=figsize)
plt.plot(loss, ls='-', color=color_train, label='loss')
plt.yscale('log')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.title('loss')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_graphRNN_loss_logscale.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# In[ ]:


print('Plot loss (part) (log scale)...')

nskip_beg = int(0.3*num_epochs)
nskip_end = 0

color_train = 'tab:blue'

figsize = figsize_lh3

plt.figure(figsize=figsize)
plt.plot(np.arange(nskip_beg, num_epochs-nskip_end), loss[nskip_beg:num_epochs-nskip_end], ls='-', color=color_train, label='loss')
plt.yscale('log')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.title('loss')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_graphRNN_loss_logscale_zoom1.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# In[ ]:


print('Plot lr...')

# Plot learning rate (lr)
# -----------------------
color_lr_G = 'green'
color_lr_E = 'orange'

figsize = figsize_lh3

plt.figure(figsize=figsize)
plt.plot(lr_used_G, ls='-', color=color_lr_G, label='lr rnn_G')
plt.plot(lr_used_E, ls='-', color=color_lr_E, label='lr rnn_E')
#plt.yscale('log')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.title('lr')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_graphRNN_lr.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# ## Generate graphs

# ### Test / Check

# In[ ]:


print('Test / check : generate graph...')

n_graph = 1
max_n_nodes = 2*data_set_n_nodes_max # should not be reached...

torch.random.manual_seed(2304)

G_gen_list, adj_seq_array_gen = generate_graph(
    rnn_G,
    rnn_E,
    max_n_nodes=max_n_nodes,
    n_graph=n_graph,
    force_node1=False,
    return_encoded=True,
    device=torch.device('cuda:0')
)


# In[ ]:


print('Test / check on generated graph...')

k = 0 # Index of generated graph

G = G_gen_list[k]                    # generated graph
adj_seq_array = adj_seq_array_gen[k] # generated encoded adjacency matrix

# Get adjacency matrix from generated graph
adj_mat_csr_1 = networkx.adjacency_matrix(G)
# Encode it
max_prev_node = rnn_G.input_size
adj_seq_array_1 = encode_adj(adj_mat_csr_1, max_prev_node=max_prev_node)

# Get adjacency matrix from generated encoded adjacency matrix
adj_mat_csr_2 = decode_adj(adj_seq_array)
# Get corresponding graph
G_2 = networkx.from_scipy_sparse_array(adj_mat_csr_2)

# Check
print('Same encoding  "adj_seq_array" ?', np.all(adj_seq_array == adj_seq_array_1))
print('Same adj. mat. "adj_mat_csr"   ?', np.all(adj_mat_csr_1.toarray() == adj_mat_csr_2.toarray()))
print('Same graph     "G"             ?', np.all(
    (np.all(np.asarray(list(G.nodes)) == np.asarray(list(G_2.nodes))),
     np.all(np.asarray(list(G.edges)) == np.asarray(list(G_2.edges))))
))


# In[ ]:


print('Test / check : plot generated graph (topology)...')

# For plotting graphs
kwds = kwds_multi.copy()

figsize = figsize_lh3
# -----

k = 0 # Index of generated graph
G = G_gen_list[k] # generated graph

# Get adjacency matrix from generated graph
adj_mat_csr = networkx.adjacency_matrix(G)
# Encode it
max_prev_node = rnn_G.input_size
adj_seq_array = encode_adj(adj_mat_csr, max_prev_node=max_prev_node)

# Plot
plt.subplots(1, 3, figsize=figsize)

plt.subplot(1, 3, 1)
networkx.draw(G, with_labels=False, **kwds)
plt.title('Generated graph')

plt.subplot(1, 3, 2)
plt.imshow(adj_mat_csr.toarray(), interpolation='none')
plt.title(f'Adjacency matrix, bw={csr_array_bw(adj_mat_csr)}')

plt.subplot(1, 3, 3)
plt.imshow(adj_seq_array, origin='upper', extent=[0.5, adj_seq_array.shape[1]+0.5, adj_seq_array.shape[0]+0.5, 0.5], interpolation='none')
# plt.title(f'Encoded adj. matrix (max_prev_node={max_prev_node})(row by row):\n node i (i-th row) is linked to prev. nodes i-j (j-th col) ?')
plt.title(f'Encoded adj. matrix\nmax_prev_node={max_prev_node}')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_gen_graph_check_{k}.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# ## Save / Export

# ### Save the data set

# In[ ]:


print('Save / export the data set (graphRNN)...')

# Save data set
with open(filename_data_set, 'wb') as f: pickle.dump(data_set, file=f)


# ### Save the model (hyper parameters and parameters)

# In[ ]:


print('Save / export the model (hyper parameters and parameters)...')

# Save model

# rnn_G 
# =====
# Hyper parameters (design of the model)
with open(filename_hyper_param_G, 'w') as f: f.write(str(rnn_G_hyper_params).replace(',', ',\n'))

# Model parameters
torch.save(rnn_G.state_dict(), filename_param_G)

# rnn_E 
# =====
# Hyper parameters (design of the model)
with open(filename_hyper_param_E, 'w') as f: f.write(str(rnn_E_hyper_params).replace(',', ',\n'))

# Model parameters
torch.save(rnn_E.state_dict(), filename_param_E)


# In[ ]:


print('Save / export loss and lr...')

# Save loss and lr
with open(filename_loss_lr, 'wb') as f: pickle.dump((loss, lr_used_G, lr_used_E), file=f)


# ## Display

# ### Display the model design

# In[ ]:


print('Display the model (graphRNN)...')

print('\n')
print('rnn_G\n-----')
print(rnn_G)
print(f'Number of (learnable) params: {nb_net_params(rnn_G)}')

print('\n')
print('rnn_E\n-----')
print(rnn_E)
print(f'Number of (learnable) params: {nb_net_params(rnn_E)}')


# ### Display the model parameters

# In[29]:


# rnn_G.state_dict() # display parameters
# rnn_E.state_dict() # display parameters

