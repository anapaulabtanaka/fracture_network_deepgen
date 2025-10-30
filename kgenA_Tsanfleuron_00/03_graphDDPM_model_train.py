#!/usr/bin/env python
# coding: utf-8

# # Model for graph nodes features generation - training

# In[1]:


import networkx
import torch, torch_geometric
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

import pyvista as pv
import os

# import json
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
# # In[3]:
# 
# 
# # Choose backend for pyvista with jupyter
# # ---------------------------------------
# # pv.set_jupyter_backend('trame')  # 3D-interactive plots
# pv.set_jupyter_backend('static') # static plots
# 
# # Notes:
# # -> ignored if run in a standard python shell
# # -> use keyword argument "notebook=False" in Plotter() to open figure in a pop-up window
# 
# 
# # ## Load local functions 
# 
# In[ ]:


print('Load local functions...')

# import sys
# sys.path.insert(1, '../utils/')

# from graph_utils import *
# from graph_ddpm import *
# from ml_utils import *
# from graph_plot import *
# # from magic_utils import *
 
with open('../utils/graph_utils.py') as f: exec(f.read())
with open('../utils/graph_ddpm.py') as f: exec(f.read())
with open('../utils/ml_utils.py') as f: exec(f.read())
with open('../utils/graph_plot.py') as f: exec(f.read())
# with open('../utils/magic_utils.py') as f: exec(f.read())


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
out_dir = 'out_graphDDPM_model' # output directory

fig_dir = 'fig'      # PARAMS

plt_show = False     # PARAMS (show graphics 2D ?)
off_screen = True    # PARAMS (show graphics 3D ?)

save_fig_png = True  # PARAMS
fig_prefix = '03'    # PARAMS

fig_counter = 0

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)

# Files for saving data set / test set (pickle) (see further)
# -----------------------------------------------------------
filename_data_set = os.path.join(out_dir, f'data_set.pickle')
filename_data_set_shift = os.path.join(out_dir, f'data_set_shift.txt')
filename_data_set_scale_factor = os.path.join(out_dir, f'data_set_scale_factor.txt')
filename_test_set = os.path.join(out_dir, f'test_set.pickle')

# Files for saving network (ddpm) (see further)
# ---------------------------------------------
filename_hyper_param_ddpm_net = os.path.join(out_dir, 'ddpm_net_hyper_params.txt')
filename_hyper_param_ddpm     = os.path.join(out_dir, 'ddpm_hyper_params.txt')

filename_param_ddpm     = os.path.join(out_dir, 'ddpm.params')

# Files for saving loss and lr (see further)
# -------------------------------------------
filename_loss_lr = os.path.join(out_dir, 'ddpm_loss_lr.pickle')



# # Data set / test set

# ### Read graphs collections

# In[ ]:


print('Read data set / test set (collection of subgraphs)...')

# Load from pickle file
data_dir = 'data_gen'
filename_graph_collection_data_set = os.path.join(data_dir, f'graph_collection_data_set.pickle')
filename_graph_collection_test_set = os.path.join(data_dir, f'graph_collection_test_set.pickle')

with open(filename_graph_collection_data_set, 'rb') as f: G_list = pickle.load(f)
with open(filename_graph_collection_test_set, 'rb') as f: G_list_test = pickle.load(f)

# Set parameters
if attr is not None:
    node_attrs      = ['pos', attr]
    node_attrs_ind  = [tuple(range(dim)), [dim + i for i in range(attr_ncomp)]]
    node_attrs_type = ['float', 'float']
else:
    node_attrs      = ['pos']
    node_attrs_ind  = [tuple(range(dim))]
    node_attrs_type = ['float']


# In[ ]:


# # print('Read data set / test set (collection of subgraphs)...')

# # Alternative - read from text files (provides the same lists)
# # ============================================================

# data_dir = 'data_gen'
# filename_data_set_basename = 'graph_collection_data_set'
# filename_test_set_basename = 'graph_collection_test_set'

# if attr is not None:
#     node_attrs      = ['pos', attr]
#     node_attrs_ind  = [tuple(range(dim)), [dim + i for i in range(attr_ncomp)]]
#     node_attrs_type = ['float', 'float']
# else:
#     node_attrs      = ['pos']
#     node_attrs_ind  = [tuple(range(dim))]
#     node_attrs_type = ['float']

# # Load graph list for data set
# G_list_2 = load_networkx_graph_list(
#     data_dir, filename_data_set_basename, 
#     suffix_nodes='_nodes.dat', 
#     suffix_edges='_links.dat', 
#     delimiter_nodes=' ',  
#     delimiter_edges=' ',
#     node_attrs=node_attrs,
#     node_attrs_ind=node_attrs_ind,
#     nodet_attrs_type=node_attrs_type,
#     start_id_at_0=True)

# # Load graph list for test set
# G_list_test_2 = load_networkx_graph_list(
#     data_dir, filename_test_set_basename, 
#     suffix_nodes='_nodes.dat', 
#     suffix_edges='_links.dat', 
#     delimiter_nodes=' ',  
#     delimiter_edges=' ',
#     node_attrs=node_attrs,
#     node_attrs_ind=node_attrs_ind,
#     nodet_attrs_type=node_attrs_type,
#     start_id_at_0=True)

# # Check - data set
# ok_list = []
# for i, (G1, G2) in enumerate(zip(G_list, G_list_2)):
#     ok = True
#     # print(f'graph #{i} ...')
#     check_list = compare_graph(G1, G2, node_attrs, verbose=0)
#     # print(f'graph #{i:3d} - check_list = {check_list}')
#     ok_list.append(np.all(check_list))

# print(f'`G_list` and `G_list_2` are equal ? {np.all(ok_list)}')
# # print(ok_list)

# # Check - test set
# ok_list = []
# for i, (G1, G2) in enumerate(zip(G_list_test, G_list_test_2)):
#     ok = True
#     # print(f'graph #{i} ...')
#     check_list = compare_graph(G1, G2, node_attrs, verbose=0)
#     # print(f'graph #{i:3d} - check_list = {check_list}')
#     ok_list.append(np.all(check_list))

# print(f'`G_list_test` and `G_list_test_2` are equal ? {np.all(ok_list)}')
# # print(ok_list)

# del(G_list_2, G_list_test_2)


# ## Convert graphs from networkx to torch_geometric and rescale features
# Each feature (coordinate of position or attribute) should be "stationary", i.e. similar statistics on every (sub)graph of the data set / test set.

# ### Convert graphs from networkx to torch_geometric
# Set ensemble of node features of interest in new features "x".
# 

# In[ ]:


print('Convert graph from data set / test set from networkx to torch_geometric...')

G_geom_list      = [torch_geometric.utils.from_networkx(G, group_node_attrs=node_attrs) for G in G_list]
G_geom_list_test = [torch_geometric.utils.from_networkx(G, group_node_attrs=node_attrs) for G in G_list_test]


# In[ ]:


print(f'Number of graphs in list for data set: {len(G_geom_list)}')
print(f'Number of graphs in list for test set: {len(G_geom_list_test)}')


# ### Compute "shift" and "scaling factor" for rescaling - on data set
# 
# Basic statistics of node features on data set: mean and variance will be used for "shift" and "scaling", see below.

# In[ ]:


print('Compute statistics (marginal) on nodes features of graphs in data set...')

# Node features on data set
node_features = [G_geom.x for G_geom in G_geom_list]

# Stats for every graph separately
node_features_min  = [x.min(axis=0).values  for x in node_features]
node_features_max  = [x.max(axis=0).values  for x in node_features]
node_features_mean = [x.mean(axis=0) for x in node_features]
node_features_var  = [x.var(axis=0)  for x in node_features]

# Mean of statistics over every graph
mean_of_min  = torch.vstack(node_features_min).mean(dim=0) 
mean_of_max  = torch.vstack(node_features_max).mean(dim=0) 
mean_of_mean = torch.vstack(node_features_mean).mean(dim=0) 
mean_of_var  = torch.vstack(node_features_var).mean(dim=0)

# Variance of mean
var_of_mean = torch.vstack(node_features_mean).var(dim=0)

# # --- or mean weighted by the number of nodes... ---
# n_nodes  = [x.shape[0] for x in node_features]
# weight = torch.tensor(n_nodes).view(-1, 1).repeat(1, 5)/torch.tensor(n_nodes).sum() # weight of each graph (repeated on each row)
# mean_of_min  = (torch.vstack(node_features_min) * weight).sum(dim=0)
# mean_of_max  = (torch.vstack(node_features_max) * weight).sum(dim=0)
# mean_of_mean = (torch.vstack(node_features_mean) * weight).sum(dim=0)
# mean_of_var  = (torch.vstack(node_features_var) * weight).sum(dim=0)
# var_of_mean = ((torch.vstack(node_features_mean)-mean_of_mean)**2 * weight).sum(dim=0)
# # ---

# Global variance
var_all = var_of_mean + mean_of_var

# Stats for all graphs together
all_graph_node_features_min  = torch.vstack(node_features).min(dim=0).values
all_graph_node_features_max  = torch.vstack(node_features).max(dim=0).values
all_graph_node_features_mean = torch.vstack(node_features).mean(dim=0)
all_graph_node_features_var  = torch.vstack(node_features).var(dim=0)
all_graph_node_features_std  = torch.vstack(node_features).std(dim=0)

print('Mean of stats on node features for every graph separately')
print('mean_of_min  = ' + ' '.join([f'{x:15.9f}' for x in mean_of_min]))
print('mean_of_max  = ' + ' '.join([f'{x:15.9f}' for x in mean_of_max]))
print('mean_of_mean = ' + ' '.join([f'{x:15.9f}' for x in mean_of_mean]))
print('mean_of_var  = ' + ' '.join([f'{x:15.9f}' for x in mean_of_var]))
print('')
print('Variance of stats on node features for every graph separately')
print('var_of_mean  = ' + ' '.join([f'{x:15.9f}' for x in var_of_mean]))
print('')
print('Stats on node features for all graphs together')
print('min_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_min]))
print('max_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_max]))
print('mean_all     = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_mean]))
print('var_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_var]))
print('std_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_std]))
print('')
print('Global variance = mean of variance + variance of mean')
print('                    [internal var. + between variance]')
print('mean_of_var + var_of_mean')
print('             = ' + ' '.join([f'{x:15.9f}' for x in var_all]))


# Set "shift" and "scaling factor" for rescaling - from data set

# In[ ]:


print('Set "shift" and "scaling factor" from data set...')

# # Based on mean of statistics for every graph separately
# # ------------------------------------------------------
# # Shift from original features to new ones (from data set)
# node_features_shift = - mean_of_mean 

# # Scaling factor from original features to new ones (from data set)
# node_features_scale_factor = 1.0 / torch.sqrt(mean_of_var)

# Based on statistics for all graphs together
# -------------------------------------------
# Shift from original features to new ones (from data set)
node_features_shift = - all_graph_node_features_mean

# Scaling factor from original features to new ones (from data set)
node_features_scale_factor = 1.0 / all_graph_node_features_std
# node_features_scale_factor = 1.0 / torch.sqrt(all_graph_node_features_var)

# Notes: 
# -----
#    mean_of_mean = all_graph_node_features_mean
#    mean_of_var = all_graph_node_features_var - var_of_mean <= all_graph_node_features_var

print(f'"shift"         : node_features_shift        = {", ".join([f"{x:.5g}" for x in node_features_shift])}')
print(f'"scaling factor": node_features_scale_factor = {", ".join([f"{x:.5g}" for x in node_features_scale_factor])}')


# ### Center and rescale node features (data set / test set)

# In[ ]:


print('Rescale (center and rescale) data set and test set...')

# Center and rescale each features of all graphs with same factor
# - data set
for G_geom in G_geom_list:
    G_geom.x = node_features_scale_factor * (G_geom.x + node_features_shift)

# - test set
for G_geom in G_geom_list_test:
    G_geom.x = node_features_scale_factor * (G_geom.x + node_features_shift)


# Basic statistics of node features on rescaled data set and rescaled test set.

# In[ ]:


print('Compute statistics (marginal) on nodes features of graphs in "rescaled" data set...')

# Node features on data set
node_features = [G_geom.x for G_geom in G_geom_list]

# Stats for every graph separately
node_features_min  = [x.min(axis=0).values  for x in node_features]
node_features_max  = [x.max(axis=0).values  for x in node_features]
node_features_mean = [x.mean(axis=0) for x in node_features]
node_features_var  = [x.var(axis=0)  for x in node_features]

# Mean of statistics over every graph
mean_of_min  = torch.vstack(node_features_min).mean(dim=0) 
mean_of_max  = torch.vstack(node_features_max).mean(dim=0) 
mean_of_mean = torch.vstack(node_features_mean).mean(dim=0) 
mean_of_var  = torch.vstack(node_features_var).mean(dim=0)

# Variance of mean
var_of_mean = torch.vstack(node_features_mean).var(dim=0)

# # --- or mean weighted by the number of nodes... ---
# n_nodes  = [x.shape[0] for x in node_features]
# weight = torch.tensor(n_nodes).view(-1, 1).repeat(1, 5)/torch.tensor(n_nodes).sum() # weight of each graph (repeated on each row)
# mean_of_min  = (torch.vstack(node_features_min) * weight).sum(dim=0)
# mean_of_max  = (torch.vstack(node_features_max) * weight).sum(dim=0)
# mean_of_mean = (torch.vstack(node_features_mean) * weight).sum(dim=0)
# mean_of_var  = (torch.vstack(node_features_var) * weight).sum(dim=0)
# var_of_mean = ((torch.vstack(node_features_mean)-mean_of_mean)**2 * weight).sum(dim=0)
# # ---

# Global variance
var_all = var_of_mean + mean_of_var

# Stats for all graphs together
all_graph_node_features_min  = torch.vstack(node_features).min(dim=0).values
all_graph_node_features_max  = torch.vstack(node_features).max(dim=0).values
all_graph_node_features_mean = torch.vstack(node_features).mean(dim=0)
all_graph_node_features_var  = torch.vstack(node_features).var(dim=0)
all_graph_node_features_std  = torch.vstack(node_features).std(dim=0)

print('Mean of stats on node features for every graph separately')
print('mean_of_min  = ' + ' '.join([f'{x:15.9f}' for x in mean_of_min]))
print('mean_of_max  = ' + ' '.join([f'{x:15.9f}' for x in mean_of_max]))
print('mean_of_mean = ' + ' '.join([f'{x:15.9f}' for x in mean_of_mean]))
print('mean_of_var  = ' + ' '.join([f'{x:15.9f}' for x in mean_of_var]))
print('')
print('Variance of stats on node features for every graph separately')
print('var_of_mean  = ' + ' '.join([f'{x:15.9f}' for x in var_of_mean]))
print('')
print('Stats on node features for all graphs together')
print('min_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_min]))
print('max_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_max]))
print('mean_all     = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_mean]))
print('var_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_var]))
print('std_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_std]))
print('')
print('Global variance = mean of variance + variance of mean')
print('                    [internal var. + between variance]')
print('mean_of_var + var_of_mean')
print('             = ' + ' '.join([f'{x:15.9f}' for x in var_all]))


# In[ ]:


print('Compute statistics (marginal) on nodes features of graphs in "rescaled" test set...')

# Node features on test set
node_features = [G_geom.x for G_geom in G_geom_list_test]

# Stats for every graph separately
node_features_min  = [x.min(axis=0).values  for x in node_features]
node_features_max  = [x.max(axis=0).values  for x in node_features]
node_features_mean = [x.mean(axis=0) for x in node_features]
node_features_var  = [x.var(axis=0)  for x in node_features]

# Mean of statistics over every graph
mean_of_min  = torch.vstack(node_features_min).mean(dim=0) 
mean_of_max  = torch.vstack(node_features_max).mean(dim=0) 
mean_of_mean = torch.vstack(node_features_mean).mean(dim=0) 
mean_of_var  = torch.vstack(node_features_var).mean(dim=0)

# Variance of mean
var_of_mean = torch.vstack(node_features_mean).var(dim=0)

# # --- or mean weighted by the number of nodes... ---
# n_nodes  = [x.shape[0] for x in node_features]
# weight = torch.tensor(n_nodes).view(-1, 1).repeat(1, 5)/torch.tensor(n_nodes).sum() # weight of each graph (repeated on each row)
# mean_of_min  = (torch.vstack(node_features_min) * weight).sum(dim=0)
# mean_of_max  = (torch.vstack(node_features_max) * weight).sum(dim=0)
# mean_of_mean = (torch.vstack(node_features_mean) * weight).sum(dim=0)
# mean_of_var  = (torch.vstack(node_features_var) * weight).sum(dim=0)
# var_of_mean = ((torch.vstack(node_features_mean)-mean_of_mean)**2 * weight).sum(dim=0)
# # ---

# Global variance
var_all = var_of_mean + mean_of_var

# Stats for all graphs together
all_graph_node_features_min  = torch.vstack(node_features).min(dim=0).values
all_graph_node_features_max  = torch.vstack(node_features).max(dim=0).values
all_graph_node_features_mean = torch.vstack(node_features).mean(dim=0)
all_graph_node_features_var  = torch.vstack(node_features).var(dim=0)
all_graph_node_features_std  = torch.vstack(node_features).std(dim=0)

print('Mean of stats on node features for every graph separately')
print('mean_of_min  = ' + ' '.join([f'{x:15.9f}' for x in mean_of_min]))
print('mean_of_max  = ' + ' '.join([f'{x:15.9f}' for x in mean_of_max]))
print('mean_of_mean = ' + ' '.join([f'{x:15.9f}' for x in mean_of_mean]))
print('mean_of_var  = ' + ' '.join([f'{x:15.9f}' for x in mean_of_var]))
print('')
print('Variance of stats on node features for every graph separately')
print('var_of_mean  = ' + ' '.join([f'{x:15.9f}' for x in var_of_mean]))
print('')
print('Stats on node features for all graphs together')
print('min_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_min]))
print('max_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_max]))
print('mean_all     = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_mean]))
print('var_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_var]))
print('std_all      = ' + ' '.join([f'{x:15.9f}' for x in all_graph_node_features_std]))
print('')
print('Global variance = mean of variance + variance of mean')
print('                    [internal var. + between variance]')
print('mean_of_var + var_of_mean')
print('             = ' + ' '.join([f'{x:15.9f}' for x in var_all]))


# ### Define the data set (rescaled) and the test set (rescaled)

# In[ ]:


print('Define the data set / test set for graphDDPM...')

# Data set
G_nsample = np.full((len(G_geom_list), ), 1) # number of times each graph in the list is sampled
data_set = Graph_geom_sampler_data_set(G_geom_list, G_nsample)

# Test set
G_nsample_test = np.full((len(G_geom_list_test), ), 1) # number of times each graph in the list is sampled
test_set = Graph_geom_sampler_data_set(G_geom_list_test, G_nsample_test)


# *Note: some graphs from the data set and test are plotted in next notebook.*

# ## Load data via a data loader, and plot first graphs

# In[ ]:


print('Define data loader...')

# Data loader (pytorch)
# ---------------------
batch_size = 6
data_loader = torch_geometric.loader.DataLoader(data_set, batch_size=batch_size, shuffle=True)
#data_loader = torch_geometric.loader.DataLoader(test_set, batch_size=batch_size, shuffle=True)


# In[ ]:


print('Plot first batches (2D)...')

torch.random.manual_seed(293) # -> for reproducibility of batches delivered by the data loader (if needed)

# 2D view 
# =======
kwds = kwds_multi.copy()

figsize = figsize_lh3
# -----

same_color_bar = False

for i_batch, G_batch in enumerate(data_loader):
    if i_batch == 3:
        break

    G_batch_geom_list = G_batch.to_data_list()    
    out_name = f'ddpm_train_set_2d_batch_{i_batch}'

    plot_graph_multi_2d_from_G_geom_list(
            G_batch_geom_list, dim,
            out_name=out_name, 
            nr=None,
            attr=attr,
            attr_label_list=attr_label_list, 
            attr_cmap_list=attr_cmap_list,
            rescale=False,
            title_list=None, title_fontsize=12,
            figsize=figsize, save_fig_png=save_fig_png, 
            filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
            with_labels=False, same_color_bar=same_color_bar, show_color_bar=True,
            show=plt_show,
            **kwds)


# In[ ]:


# %%skip_if dim == 2
if dim == 3:
    print('Plot first batches (3D)...')
    
    torch.random.manual_seed(293) # -> for reproducibility of batches delivered by the data loader (if needed)

    # 3D view 
    # =======
    kwargs_edges = kwargs_edges_multi.copy()
    kwargs_pts = kwargs_pts_multi.copy()
    kwargs_pts_labels = kwargs_pts_labels_multi.copy()
    kwargs_scalar_bar = kwargs_scalar_bar_multi.copy()

    window_size = [int(0.66*x) for x in window_size_multi]
    # -----

    notebook = True  # inline
    cpos = None

    same_color_bar = False

    for i_batch, G_batch in enumerate(data_loader):
        if i_batch == 3:
            break

        G_batch_geom_list = G_batch.to_data_list()
        out_name = f'ddpm_train_set_3d_batch_{i_batch}'

        plot_graph_multi_3d_from_G_geom_list(
                G_batch_geom_list, dim,
                out_name=out_name, 
                nr=None,
                attr=attr,
                attr_label_list=attr_label_list, 
                attr_cmap_list=attr_cmap_list,
                rescale=False,
                title_list=None, title_fontsize=12,
                notebook=notebook, window_size=window_size, save_fig_png=save_fig_png, off_screen=off_screen,
                filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
                with_labels=False, same_color_bar=same_color_bar, show_color_bar=True,
                kwargs_edges=kwargs_edges, kwargs_pts=kwargs_pts, kwargs_scalar_bar=kwargs_scalar_bar, kwargs_pts_labels=kwargs_pts_labels,
                cpos=cpos, print_cpos=False)


# In[20]:


if save_fig_png:
    fig_counter = fig_counter+1


# ## Split the data set into training set and validation set by random sampling

# In[ ]:


print('Split the data set into training set and validation set (random sampling)...')

# Split data set into training set and validation set
seed = 234
valid_frac = 0.2

n = len(data_set.G_geom_list)
G_geom_list = [data_set.G_geom_list[i] for i in torch.randperm(n)]

n_valid = int(valid_frac*n)
n_train = n - n_valid

train_set = Graph_geom_sampler_data_set(G_geom_list[:n_train], np.full((n_train, ), 1))
valid_set = Graph_geom_sampler_data_set(G_geom_list[n_train:], np.full((n_valid, ), 1))


# ## DDPM model for graph node features generation

# ### Define the model (design)

# In[ ]:


print('Define the model (design)...')

# Model
# -----

# Number of time steps
# n_steps = 2000
n_steps = 2400
# n_steps = 5000

# Embedding dimension for time steps
time_emb_dim = 64 #100 #64 #200

# Noise schedule

# # - constant
# betas = 1.e-3

# # - linear
# # betas = np.linspace(1.e-4, 2.e-2, n_steps).tolist()
# betas = np.linspace(1.e-4, 1.e-2, n_steps).tolist()

# # - cosine (https://arxiv.org/pdf/2102.09672.pdf)
# beta_clip_min, beta_clip_max = 1.e-4, 5.e-2 # 0.999
# epsilon = 8.e-3
# steps = torch.arange(n_steps+1).to(torch.float32)
# f_t = torch.cos(((steps/n_steps + epsilon) / (1.0 + epsilon))*torch.pi*0.5)**2
# betas = torch.clip(1.0-f_t[1:]/f_t[:n_steps], beta_clip_min, beta_clip_max).tolist()

# - cosine (https://arxiv.org/pdf/2102.09672.pdf)
m = 100
n = n_steps - m
beta_clip_min, beta_clip_max = 1.e-3, 1.e-2 #5.e-4, 1.2e-2 #1.e-4, 2.e-2 #5.e-4, 1.2e-2
epsilon = 8.e-3
steps = torch.arange(n+1).to(torch.float32)
f_t = torch.cos(((steps/n + epsilon) / (1.0 + epsilon))*torch.pi*0.5)**2
betas = torch.clip(1.0-f_t[1:]/f_t[:n], beta_clip_min, beta_clip_max).tolist()
betas = betas + m*[betas[-1]]

# Hyper parameters (design of the model)
# ddpm_net_hyper_params = dict(
#     n_node_features = data_set.n_node_features,
#     nf_list         = [20, 40, 80, 100, 200, 300],
#     nf_last         = 20,
#     has_mid         = False,
#     activation      = torch.nn.SiLU(),
#     te_activation   = torch.nn.SiLU(),
#     n_steps         = n_steps, 
#     time_emb_dim    = time_emb_dim
# )

#nf_list = (np.array([10, 20, 40, 80, 160])*data_set.n_node_features).tolist()
#nf_list = (np.full(5, 10)*data_set.n_node_features).tolist()
# nf_list = (np.array([20, 10, 8, 6, 4])*data_set.n_node_features).tolist()
# nf_list = (np.array([20, 15, 10, 8, 6, 4, 2])*data_set.n_node_features).tolist()
# nf_list = (np.array([20, 10, 10, 8, 8, 4, 4])*data_set.n_node_features).tolist()
#nf_list = (np.array([32, 16, 16, 12, 12, 8, 8, 4, 4])*data_set.n_node_features).tolist()

nf_list = (np.array([64, 32, 32, 16, 16, 8, 8, 4, 4])*data_set.n_node_features).tolist()

ddpm_net_hyper_params = dict(
    n_node_features = data_set.n_node_features,
    # nf_list         = (np.array([2, 4, 4, 8, 8, 16, 16, 32])*data_set.n_node_features).tolist(),
    nf_list         = nf_list,
    nf_last         = None,
    has_mid         = True, #True, #False,
    nf_mid          = nf_list[-2], #None,
    activation      = torch.nn.LeakyReLU(), #torch.nn.ReLU(), #torch.nn.SiLU(),
    te_activation   = torch.nn.LeakyReLU(), #torch.nn.ReLU(), #torch.nn.SiLU(),
    normalize_down  = True,
    op_down1        = 'SAGEConv', #'ResGatedGraphConv', #'GATConv', #'GraphConv', #'SAGEConv', #'GCNConv',
    op_down2        = 'SAGEConv', #None, #'GraphConv',
    normalize_up    = True,
    op_up1          = 'SAGEConv',
    op_up2          = 'SAGEConv', #None, #'GraphConv',
    normalize_mid   = True,
    op_mid1         = 'SAGEConv',
    op_mid2         = 'SAGEConv', #None, #'GraphConv',
    normalize_last  = False,
    op_last         = 'Linear', #'Linear', #'GraphConv',
    n_steps         = n_steps, 
    time_emb_dim    = time_emb_dim
)

ddpm_hyper_params = dict(
    n_node_features = data_set.n_node_features,
    n_steps         = n_steps,
    betas           = betas
)

ddpm = Graph_DDPM(Graph_DDPM_net_model(**ddpm_net_hyper_params), **ddpm_hyper_params)

# nb_net_params(ddpm)


# ### Load the model (hyper parameters and parameters) - if existing and already trained

# In[ ]:


# # print('Load the model (hyper parameters and parameters) (graphDDPM)...')

# # Define activation function as they appear in hyper parameters...
# from torch.nn import LeakyReLU, ReLU, SiLU, Sigmoid, Tanh

# # Load model

# # Hyper parameters (design of the model)
# with open(filename_hyper_param_ddpm_net, 'r') as f: ddpm_net_hyper_params = eval(f.read())
# with open(filename_hyper_param_ddpm, 'r') as f: ddpm_hyper_params = eval(f.read())

# # # Hyper parameters (design of the model)
# # with open(filename_hyper_param_ddpm_net, 'r') as f: ddpm_net_hyper_params = json.load(f)
# # with open(filename_hyper_param_ddpm, 'r') as f: ddpm_hyper_params = json.load(f)

# # Model (parameters)
# ddpm = Graph_DDPM(Graph_DDPM_net_model(**ddpm_net_hyper_params), **ddpm_hyper_params)
# ddpm.load_state_dict(torch.load(filename_param_ddpm))


# In[ ]:


# # print('Load loss and lr...')

# # Load loss and lr
# with open(filename_loss_lr, 'rb') as f: train_loss, valid_loss, lr_used = pickle.load(file=f)


# ### Display the model design

# In[ ]:


print('Display the model (graphDDPM)...')

print(ddpm)
print(f'Number of (learnable) params: {nb_net_params(ddpm)}')


# ### Display the model parameters

# In[26]:


# ddpm.state_dict() # display parameters


# ## Some check on the model

# ### Noise schedule and variance of features as function of time step

# In[ ]:


print('Do some check (plot): noise schedule and variance of features as function of time step...')

# Noise schedule and variance of features after each time step

figsize = figsize_lh3

plt.subplots(1,2, figsize=figsize)

plt.subplot(1,2,1)
plt.plot(ddpm.betas.to('cpu').numpy())
#plt.yscale('log')
plt.grid()
plt.title('Noise schedule (betas)')

plt.subplot(1,2,2)
plt.plot(1.0 - ddpm.alpha_bars.to('cpu').numpy())
plt.grid()
plt.title(f'Variance (1 - alpha_bars, last = {1.0 - ddpm.alpha_bars[-1]:.5})')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_ddpm_schedule.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# ### Features after all time steps VS normal N(0,1)

# In[ ]:


print('Do some check (plot): features after all time steps VS N(0,1)...')

# Check that the noise (on graph node features), after all steps of forward process, follows a normal distribution N(0, 1)

d_set = train_set
#d_set = data_set

# QQ-plot of distribution of graph node feature (Zi) vs N(0, 1)

# n = len(set)
# batch = next(iter(torch_geometric.loader.DataLoader(d_set, batch_size=n, shuffle=False)))

batch_size = 50
data_loader = torch_geometric.loader.DataLoader(d_set, batch_size=batch_size, shuffle=False)

device = 'cuda:0'
ddpm.to_device(device)
Z = []
for G_batch in data_loader:
    G_batch = ddpm.forward(G_batch.to(device), torch.full((G_batch.num_graphs, ), ddpm.n_steps-1, device=device))
    Z.append(G_batch.x.to('cpu').numpy())
ddpm.to_device('cpu')

Z = np.vstack(Z) # one features per column

q = np.linspace(.001, .999, 100)
Zq = np.quantile(Z, q=q, axis=0) # Zq[i, j] : q[i]-quantile of Z[:, j]
Nq = scipy.stats.norm.ppf(q)

nr = Zq.shape[1]
nc = 2

figsize = (figsize_lh3[0], figsize_lh3[1]*nr*.75)

plt.subplots(nr, nc, figsize=figsize)

for j in range(nr):
    plt.subplot(nr, nc, j*nc+1)
    plt.plot(Zq[:, j], Nq, ls='', marker='.')
    mi, ma = Zq[:, j].min(), Zq[:, j].max()
    plt.plot([mi, ma], [mi, ma], ls='dashed')
    plt.grid()
    plt.xlabel(f'Z{j}')
    plt.ylabel('N(0,1)')
    if j==0:
        plt.title('QQ-plot')

    plt.subplot(nr, nc, j*nc+2)
    plt.hist(Z[:, j], bins=50, density=True, label=f'Z{j}')
    t = np.linspace(Z[:, j].min(), Z[:, j].max(), 300)
    plt.plot(t, scipy.stats.norm.pdf(t), ls='dashed', label='N(0,1)')
    plt.axvline(x=np.mean(Z[:, j]), ls='dashed', color='purple', label='mean')
    plt.grid()
    plt.xlabel(f'Z{j}')
    plt.legend(fontsize=8)
    if j==0:
        plt.title('Density')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_ddpm_check_noise.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# In[ ]:


# QQ-plot of distribution of all graph node features (Z_all) vs N(0, 1)

Z_all = Z.reshape(-1)

#q = np.linspace(.001, .999, 100)
Zq = np.quantile(Z_all, q=q)
Nq = scipy.stats.norm.ppf(q)

figsize = figsize_lh3

plt.subplots(1,2,figsize=figsize)

plt.subplot(1,2,1)
plt.plot(Zq, Nq, ls='', marker='.')
plt.plot([Zq.min(), Zq.max()], [Zq.min(), Zq.max()], ls='dashed')
plt.grid()
plt.xlabel('Z (all)')
plt.ylabel('N(0,1)')
plt.title('QQ-plot')

plt.subplot(1,2,2)
plt.hist(Zq, bins=50, density=True, label='Z')
t = np.linspace(Z_all.min(), Z_all.max(), 300)
plt.plot(t, scipy.stats.norm.pdf(t), ls='dashed', label='N(0,1)')
plt.axvline(x=np.mean(Z_all), ls='dashed', color='purple', label='mean')
plt.grid()
plt.xlabel('Z (all)')
plt.legend()
plt.title('Density')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_ddpm_check_noise_all_features_grouped.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# *Note: diffusion of one graph is illustrated in next notebook.*

# ## Training

# ### Other model settings

# In[ ]:


print('Set loss function...')

# Loss function
# -------------
loss_func = torch.nn.MSELoss()


# ### Re-initialize the model parameters (if needed)

# In[ ]:


print('[Re-]initialize the model parameters...')

# Initialize the network parameters
# ---------------------------------
ddpm_seed = 903
torch.random.manual_seed(ddpm_seed)
#ddpm.net.init_weights(gain=0.1)
reset_all_parameters(ddpm)
    
# Initialize lists for storing loss, lr
# -------------------------------------
train_loss, valid_loss = [], []
lr_used = []


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
num_epochs = 4001
print_epoch = 1

# Optimizer
# ---------------------
# # - 1. Stochastic Gradient Descent (SGD)
# lr = .001             # learning rate
# weight_decay = 0.00 # L2-regularization
# momentum = 0.00     # momentum
# optimizer = torch.optim.SGD(ddpm.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
# - 2. Adam
lr = 0.002            # learning rate; default: lr=0.001
adam_betas = (0.9, 0.999)    # betas parameters; default: betas=(0.9, 0.999)
# lr = 0.0002           # learning rate; default: lr=0.001
# adam_betas = (0.5, 0.999)    # betas parameters; default: betas=(0.9, 0.999)
eps = 1.e-8           # epsilon parameter; default: 1.e-8
weight_decay = 0.000   # L2-regularization; default: 0.0
optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr, betas=adam_betas, eps=eps, weight_decay=weight_decay)
# ...

# Learning rate scheduler
# -----------------------
# # - 1. CosineAnnealingLR
# lr_init = lr
# T_max = num_epochs / 7. # num_epochs
# eta_min = lr_init / 5.
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=-1, verbose=False)
# # Essentially, with eta_max = lr_init, at epoch t, the learning rate is set to
# #    eta_t = eta_min + 1/2 * (eta_max - eta_min) * (1 + cos(t/T_max*pi))
#
# - 2. CosineAnnealingWarmRestarts
lr_init = lr
eta_min = lr_init / 10.
T_mult = 2
nrestart = 3
T_0 = int(np.ceil(num_epochs * (T_mult-1) / (T_mult**(nrestart+1)-1)))
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=-1, verbose=False)
# Essentially, with eta_max = lr_init, at epoch t, the learning rate is set to
#    eta_t = eta_min + 1/2 * (eta_max - eta_min) * (1 + cos(T_cur/T_i*pi))
# where
#    T_cur: the number of epochs since the last restart
#    T_i  : the number of epochs between two warm restarts
#    T_0  : number of epochs before the 1st warm restart
#    T_mult: T_i is defined as T_i = T_mult * T_{i-1}
#
# # - 3. MultiStepLR
# gamma = .3
# milestones = [int(num_epochs/9), int(num_epochs/3)]
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=-1, verbose=False)
# # At epoch in `milestones`, lr is multiplied by `gamma`
#
# # - 4.
# lr_scheduler = None
# ...

# ----
# # Note: to compute the sequence of lr used, and plot it: 
# lr_used = []
# for i in range(num_epochs):
#     lr_used.append(lr_scheduler.get_last_lr()[0])
#     optimizer.step()
#     lr_scheduler.step()

# plt.figure()
# plt.plot(lr_used)
# plt.xlabel('epoch')
# plt.grid()
# plt.show()
# ========

# ---
# Fixed batch of graph with random noise for generating data to save during training
ng_fixed = 12 # number of graphs
torch.manual_seed(983)

# # Set G_batch_fixed
# G_batch_1 = next(iter(torch_geometric.loader.DataLoader(train_set, batch_size=ng_fixed//2, shuffle=True)))
# G_batch_2 = next(iter(torch_geometric.loader.DataLoader(valid_set, batch_size=ng_fixed - ng_fixed//2, shuffle=True)))
# G_batch_fixed = torch_geometric.data.Batch.from_data_list(G_batch_1.to_data_list() + G_batch_2.to_data_list())

# Set G_batch_fixed
G_batch_fixed_initial = next(iter(torch_geometric.loader.DataLoader(test_set, batch_size=12, shuffle=True)))
G_batch_fixed = G_batch_fixed_initial.clone()

# # Set G_batch_fixed
# G_batch_fixed = next(iter(torch_geometric.loader.DataLoader(data_set, batch_size=12, shuffle=True)))

torch.manual_seed(878)
G_batch_fixed.x = torch.randn_like(G_batch_fixed.x)
save_gen_epoch = 200 # save generated data at every `save_epoch`
save_gen_dir = out_dir + '/train_intermediate'
save_gen_file_fmt= save_gen_dir + '/ddpm_{:04d}.pt' # where to save generated data
if not os.path.isdir(save_gen_dir):
    os.mkdir(save_gen_dir)
# ---

# Create Data Loader for training data set
train_data_loader = torch_geometric.loader.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Create Data Loader for validation data set
valid_batch_size = 100
valid_data_loader = torch_geometric.loader.DataLoader(valid_set, batch_size=valid_batch_size, shuffle=False)
# valid_data_loader = None

# Train
t1 = time.time()
train_loss_cur, valid_loss_cur, lr_used_cur = \
    train_graph_ddpm(
        train_data_loader, 
        ddpm, 
        optimizer,
        loss_func,
        lr_scheduler=lr_scheduler,
        return_lr=True,
        return_loss=True,
        num_epochs=num_epochs,
        valid_data_loader=valid_data_loader,
        print_epoch=print_epoch,
        G_batch_fixed=G_batch_fixed,
        save_gen_epoch=save_gen_epoch,
        save_gen_file_fmt=save_gen_file_fmt,
        # device=torch.device('cpu')
        device=torch.device('cuda:0')
    )
t2 = time.time()

# Update lists of loss, accuracy, score, lr
train_loss = train_loss + train_loss_cur # concatenate list
valid_loss = valid_loss + valid_loss_cur # concatenate list

lr_used = lr_used + lr_used_cur # concatenate list

# Print elapsed time and result of last epoch
print(f'Elapsed time for {num_epochs} epochs: {t2-t1:.3g} s')
print(f'Last epoch, loss : train: {train_loss[-1]:.2g}, valid: {valid_loss[-1]:.2g}')


# In[ ]:


print('Plot loss...')

# Plot loss (training and validation)
# -----------------------------------
color_train = 'tab:blue'
color_valid = 'tab:red'

figsize = figsize_lh3

plt.figure(figsize=figsize)
# loss
plt.plot(train_loss, ls='-', color=color_train, label='train loss')
plt.plot(valid_loss, ls='-', color=color_valid, label='valid loss')
#plt.yscale('log')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.title('loss')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_ddpm_loss.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# In[ ]:


print('Plot loss (log scale)...')

# Plot loss (training and validation) (log scale along y axis)
# -----------------------------------
color_train = 'tab:blue'
color_valid = 'tab:red'

figsize = figsize_lh3

plt.figure(figsize=figsize)
# loss
plt.plot(train_loss, ls='-', color=color_train, label='train loss')
plt.plot(valid_loss, ls='-', color=color_valid, label='valid loss')
plt.yscale('log')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.title('loss')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_ddpm_loss_logscale.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# In[ ]:


print('Plot loss (part) (log scale)...')

nskip_beg = int(0.3*num_epochs)
nskip_end = 0

figsize = figsize_lh3

plt.figure(figsize=figsize)
# loss
plt.plot(np.arange(nskip_beg, num_epochs-nskip_end), train_loss[nskip_beg:num_epochs-nskip_end], ls='-', color=color_train, label='train loss')
plt.plot(np.arange(nskip_beg, num_epochs-nskip_end), valid_loss[nskip_beg:num_epochs-nskip_end], ls='-', color=color_valid, label='valid loss')
plt.yscale('log')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.title('loss')

if save_fig_png:
    plt.tight_layout()
    fig_counter = fig_counter-1
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_ddpm_loss_logscale_zoom1.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# In[35]:


# train_loss_S = [t for t in train_loss]
# valid_loss_S = [t for t in valid_loss]


# In[ ]:


print('Plot lr...')

# Plot learning rate (lr)
# -----------------------
color_lr = 'orange'

figsize = figsize_lh3

plt.figure(figsize=figsize)
plt.plot(lr_used, ls='-', color=color_lr, label='lr ddpm')
#plt.yscale('log')
plt.xlabel('epoch')
plt.legend()
plt.grid()
plt.title('lr')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_ddpm_lr.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# ### Show data generated during training

# In[ ]:


# subplots
mr = 1                  # nb of line
mc = min(8, ng_fixed)   # nb of cols (graph in one line)


# In[ ]:


print('Plot data generated during training (2D)...')

# 2D view 
# =======
kwds = kwds_multi_line.copy()

figsize = figsize_multi_line
# -----

# Plot generated data at given epoch, saved during training
# ---------------------------------------------------------
epoch_list = list(range(0, num_epochs, save_gen_epoch))

for epoch in epoch_list:
    G_batch = torch.load(save_gen_file_fmt.format(epoch))
    G_batch_geom_list = G_batch.to_data_list()[:mc]
    out_name = f'ddpm_train_2d_after_epoch_{epoch}'

    plot_graph_multi_2d_from_G_geom_list(
            G_batch_geom_list, dim,
            out_name=out_name, 
            nr=1,
            attr=attr,
            attr_label_list=attr_label_list, 
            attr_cmap_list=attr_cmap_list,
            rescale=False,
            title_list=None, title_fontsize=8,
            figsize=figsize, save_fig_png=save_fig_png, 
            filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
            with_labels=False, same_color_bar=True, show_color_bar=True,
            show=plt_show,
            **kwds)


# In[ ]:


# Comparison with graph of fixed batch
# ------------------------------------
G_batch_geom_list = G_batch_fixed_initial.to_data_list()[:mc]
out_name = f'ddpm_train_2d_initial_graphs'

plot_graph_multi_2d_from_G_geom_list(
        G_batch_geom_list, dim,
        out_name=out_name, 
        nr=1,
        attr=attr,
        attr_label_list=attr_label_list, 
        attr_cmap_list=attr_cmap_list,
        rescale=False,
        title_list=None, title_fontsize=8,
        figsize=figsize, save_fig_png=save_fig_png, 
        filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
        with_labels=False, same_color_bar=True, show_color_bar=True,
        show=plt_show,
        **kwds)


# In[ ]:


# %%skip_if dim == 2
if dim == 3:
    print('Plot data generated during training (3D)...')

    # 3D view 
    # =======
    kwargs_edges = kwargs_edges_multi_line.copy()
    kwargs_pts = kwargs_pts_multi_line.copy()
    kwargs_pts_labels = kwargs_pts_labels_multi_line.copy()
    kwargs_scalar_bar = kwargs_scalar_bar_multi_line.copy()

    window_size = window_size_multi_line
    # -----

    notebook = True  # inline
    cpos = None

    # Plot generated data at given epoch, saved during training
    # ---------------------------------------------------------
    epoch_list = list(range(0, num_epochs, save_gen_epoch))

    for epoch in epoch_list:
        G_batch = torch.load(save_gen_file_fmt.format(epoch))
        G_batch_geom_list = G_batch.to_data_list()[:mc]
        out_name = f'ddpm_train_3d_after_epoch_{epoch}'

        plot_graph_multi_3d_from_G_geom_list(
                G_batch_geom_list, dim,
                out_name=out_name, 
                nr=1,
                attr=attr,
                attr_label_list=attr_label_list, 
                attr_cmap_list=attr_cmap_list,
                rescale=False,
                title_list=None, title_fontsize=8,
                notebook=notebook, window_size=window_size, save_fig_png=save_fig_png, off_screen=off_screen,
                filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
                with_labels=False, same_color_bar=True, show_color_bar=True,
                kwargs_edges=kwargs_edges, kwargs_pts=kwargs_pts, kwargs_scalar_bar=kwargs_scalar_bar, kwargs_pts_labels=kwargs_pts_labels,
                cpos=cpos, print_cpos=False)


# In[ ]:


# %%skip_if dim == 2
if dim == 3:

    notebook = True  # inline
    cpos = None

    # Comparison with graph of fixed batch
    # ------------------------------------
    G_batch_geom_list = G_batch_fixed_initial.to_data_list()[:mc]
    out_name = f'ddpm_train_3d_initial_graphs'

    plot_graph_multi_3d_from_G_geom_list(
            G_batch_geom_list, dim,
            out_name=out_name, 
            nr=1,
            attr=attr,
            attr_label_list=attr_label_list, 
            attr_cmap_list=attr_cmap_list,
            rescale=False,
            title_list=None, title_fontsize=8,
            notebook=notebook, window_size=window_size, save_fig_png=save_fig_png, off_screen=off_screen,
            filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
            with_labels=False, same_color_bar=True, show_color_bar=True,
            kwargs_edges=kwargs_edges, kwargs_pts=kwargs_pts, kwargs_scalar_bar=kwargs_scalar_bar, kwargs_pts_labels=kwargs_pts_labels,
            cpos=cpos, print_cpos=False)


# In[42]:


if save_fig_png:
    fig_counter = fig_counter+1


# ## Save / Export

# ### Save the data set / test set

# In[ ]:


print('Save / export the data set / test set (graphDDPM)...')

# Save data set
with open(filename_data_set, 'wb') as f: pickle.dump(data_set, file=f)
with open(filename_data_set_shift, 'w')  as f: np.savetxt(f, node_features_shift)
with open(filename_data_set_scale_factor, 'w')  as f: np.savetxt(f, node_features_scale_factor)

# Save test set
with open(filename_test_set, 'wb') as f: pickle.dump(test_set, file=f)


# ### Save the model (hyper parameters and parameters)

# In[ ]:


print('Save / export the model (hyper parameters and parameters)...')

# Save model

# # Hyper parameters (design of the model)
# with open(filename_hyper_param_ddpm_net, 'w') as f: json.dump(ddpm_net_hyper_params, f)
# with open(filename_hyper_param_ddpm, 'w')     as f: json.dump(ddpm_hyper_params, f)

# Hyper parameters (design of the model)
with open(filename_hyper_param_ddpm_net, 'w') as f: f.write(str(ddpm_net_hyper_params).replace(',', ',\n'))
with open(filename_hyper_param_ddpm, 'w')     as f: f.write(str(ddpm_hyper_params).replace(',', ',\n'))

# Model parameters
torch.save(ddpm.state_dict(), filename_param_ddpm)


# In[ ]:


print('Save / export loss and lr...')

# Save loss and lr
with open(filename_loss_lr, 'wb') as f: pickle.dump((train_loss, valid_loss, lr_used), file=f)


# ## Display

# ### Display the model design

# In[ ]:


print('Display the model (graphDDPM)...')

print(ddpm)
print(f'Number of (learnable) params: {nb_net_params(ddpm)}')


# ### Display the model parameters

# In[47]:


# ddpm.state_dict() # display parameters

