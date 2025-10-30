#!/usr/bin/env python
# coding: utf-8

# # Graph generation with node features

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
# from graph_rnn import *
# from graph_ddpm import *
# from ml_utils import *
# from graph_plot import *
# # from magic_utils import *
 
with open('../utils/graph_utils.py') as f: exec(f.read())
with open('../utils/graph_rnn.py') as f: exec(f.read())
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
# For saving generated graphs.

# In[ ]:


print('Define output settings...')

# Output directory (for saving)
# -----------------------------
out_dir = 'out_gen_graph' # output directory

fig_dir = 'fig'      # PARAMS

plt_show = False     # PARAMS (show graphics 2D ?)
off_screen = True    # PARAMS (show graphics 3D ?)

save_fig_png = True  # PARAMS
fig_prefix = '05'    # PARAMS

fig_counter = 0

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)

# Files for saving generated graphs (pickle / txt) (see further)
# --------------------------------------------------------------
filename_gen_graph_pk = os.path.join(out_dir, f'gen_graph_list.pickle')
filename_gen_graph_basename_txt = 'gen_graph_list'


# ## Load the models (graphRNN and graphDDPM)
# 

# In[ ]:


print('Load the models (graphRNN and graphDDPM)...')
 
# RNNs models
# -----------
in_dir_rnn = 'out_graphRNN_model' # input directory

if not os.path.isdir(in_dir_rnn):
    print('ERROR: no input directory (rnn model)')

filename_hyper_param_G = os.path.join(in_dir_rnn, 'rnn_G_hyper_params.txt')
filename_hyper_param_E = os.path.join(in_dir_rnn, 'rnn_E_hyper_params.txt')
filename_param_G = os.path.join(in_dir_rnn, 'rnn_G.params')
filename_param_E = os.path.join(in_dir_rnn, 'rnn_E.params')

# DDPM model
# ----------
in_dir_ddpm = 'out_graphDDPM_model' # input directory

if not os.path.isdir(in_dir_ddpm):
    print('ERROR: no input directory (ddpm model)')

filename_hyper_param_ddpm_net = os.path.join(in_dir_ddpm, 'ddpm_net_hyper_params.txt')
filename_hyper_param_ddpm     = os.path.join(in_dir_ddpm, 'ddpm_hyper_params.txt')

filename_param_ddpm     = os.path.join(in_dir_ddpm, 'ddpm.params')

# node features shift and scale factor from data set
filename_data_set_ddpm_shift = os.path.join(in_dir_ddpm, f'data_set_shift.txt')
filename_data_set_ddpm_scale_factor = os.path.join(in_dir_ddpm, f'data_set_scale_factor.txt')


# ### Load the RNN models (hyper parameters and parameters)

# In[ ]:


# Load model

# rnn_G 
# =====
# Hyper parameters (design of the model)
with open(filename_hyper_param_G, 'r') as f: rnn_G_hyper_params = eval(f.read())

# RNN model (parameters)
rnn_G = RNN_model(**rnn_G_hyper_params)
rnn_G.load_state_dict(torch.load(filename_param_G))

# rnn_E 
# =====
# Hyper parameters (design of the model)
with open(filename_hyper_param_E, 'r') as f: rnn_E_hyper_params = eval(f.read())

# RNN model (parameters)
rnn_E = RNN_model(**rnn_E_hyper_params)
rnn_E.load_state_dict(torch.load(filename_param_E))


# ### Load the DDPM model (hyper parameters and parameters)

# In[ ]:


# Define activation function as they appear in hyper parameters...
from torch.nn import LeakyReLU, ReLU, SiLU, Sigmoid, Tanh

# Load model

# # Hyper parameters (design of the model)
# with open(filename_hyper_param_ddpm_net, 'r') as f: ddpm_net_hyper_params = json.load(f)
# with open(filename_hyper_param_ddpm, 'r') as f: ddpm_hyper_params = json.load(f)

# Hyper parameters (design of the model)
with open(filename_hyper_param_ddpm_net, 'r') as f: ddpm_net_hyper_params = eval(f.read())
with open(filename_hyper_param_ddpm, 'r')     as f: ddpm_hyper_params = eval(f.read())


# Model (parameters)
ddpm = Graph_DDPM(Graph_DDPM_net_model(**ddpm_net_hyper_params), **ddpm_hyper_params)
ddpm.load_state_dict(torch.load(filename_param_ddpm))


# In[10]:


# Load node features shift and scale factor
with open(filename_data_set_ddpm_shift, 'r') as f: node_features_shift = np.loadtxt(f)
with open(filename_data_set_ddpm_scale_factor, 'r') as f: node_features_scale_factor = np.loadtxt(f)

node_features_shift_inv = - node_features_shift
node_features_scale_factor_inv = 1.0 / node_features_scale_factor

n_node_features = len(node_features_scale_factor)


# ### Display the RNN models design

# In[ ]:


# print('Display the model (graphRNN)...')

# print('\n')
# print('rnn_G\n-----')
# print(rnn_G)
# print(f'Number of (learnable) params: {nb_net_params(rnn_G)}')

# print('\n')
# print('rnn_E\n-----')
# print(rnn_E)
# print(f'Number of (learnable) params: {nb_net_params(rnn_E)}')


# ### Display the DDPM model design

# In[ ]:


# print('Display the model (graphDDPM)...')

# print(ddpm)
# print(f'Number of (learnable) params: {nb_net_params(ddpm)}')


# ## Generate graphs

# ### Generate topology

# In[ ]:


# print('Generate graphs - topology (using graphRNN model)...')

# n_graph = 500
# max_n_nodes = 10000 # should not be reached...
# # max_n_nodes = 200

# torch.random.manual_seed(2304)

# t1 = time.time()
# G_gen_list = generate_graph(
#     rnn_G,
#     rnn_E,
#     max_n_nodes=max_n_nodes,
#     n_graph=n_graph,
#     force_node1=True,
#     return_encoded=False,
#     device=torch.device('cuda:0')
# )
# t2 = time.time()
# print(f'Elapsed time for generating {n_graph} graph(s) - topology: {t2-t1:.3g} s')


# In[ ]:


print('Generate graphs - topology (using graphRNN model)...')

n_graph = 500

max_n_nodes = 10000 # should not be reached...
# max_n_nodes = 200
min_n_nodes = 5 # will re-draw graph(s) if fewer nodes

torch.random.manual_seed(2304)

t1 = time.time()
G_gen_list = generate_graph_min_n_nodes(
    rnn_G,
    rnn_E,
    min_n_nodes=min_n_nodes,
    max_n_nodes=max_n_nodes,
    n_graph=n_graph,
    force_node1=False,
    return_encoded=False,
    device=torch.device('cuda:0')
)
t2 = time.time()
print(f'Elapsed time for generating {n_graph} graph(s): {t2-t1:.3g} s')


# ### Generate node features

# In[ ]:


print('Generate node features (using graphDDPM model)...')

# Generate node features (position) (inplace)
torch.random.manual_seed(214)

end_rescale = node_features_scale_factor_inv
end_center = node_features_shift_inv

t1 = time.time()
G_gen_list = generate_list_graph_node_features(
                G_gen_list, 
                ddpm, 
                attr='x', 
                end_rescale=end_rescale,
                end_center=end_center, 
                device=torch.device('cuda:0'))
t2 = time.time()

print(f'Elapsed time for generating {n_graph} graph(s) - node features: {t2-t1:.3g} s')


# In[ ]:


print('Set networkx representation of generated graphs with position and other attributes separated...')

# Set networkx representation of generated graphs with position and other attributes separated
for G in G_gen_list:
    v = np.asarray(list(networkx.get_node_attributes(G, 'x').values()))
    remove_node_attribute(G, 'x')

    dict_pos = {i:vi[:dim].tolist() for i, vi in enumerate(v)}
    networkx.set_node_attributes(G, dict_pos, 'pos')

    if attr is not None:
        dict_attr = {i:vi[dim:].tolist() for i, vi in enumerate(v)}
        networkx.set_node_attributes(G, dict_attr, attr)
        


# ### Show first generated graphs

# In[ ]:


out_name = 'gen_graphs'


# In[ ]:


print('Plot first generated graphs (2D)...')

# 2D view 
# =======
kwds = kwds_multi.copy()

figsize = figsize_multi
# -----

ng = 16
ng = min(len(G_gen_list), ng)

same_color_bar = False

plot_graph_multi_2d_from_G_networkx_list(
        G_gen_list[:ng], 
        out_name=out_name, 
        nr=None,
        attr=attr,
        attr_label_list=attr_label_list, 
        attr_cmap_list=attr_cmap_list,
        title_list=None, title_fontsize=12,
        figsize=figsize, save_fig_png=save_fig_png, 
        filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
        with_labels=False, same_color_bar=same_color_bar, show_color_bar=True,
        show=plt_show,
        **kwds)


# In[ ]:


# %%skip_if dim == 2
if dim == 3:
    print('Plot first generated graphs (3D)...')

    # 3D view 
    # =======
    kwargs_edges = kwargs_edges_multi.copy()
    kwargs_pts = kwargs_pts_multi.copy()
    kwargs_pts_labels = kwargs_pts_labels_multi.copy()
    kwargs_scalar_bar = kwargs_scalar_bar_multi.copy()

    window_size = window_size_multi
    # -----

    ng = 9
    ng = min(len(G_gen_list), ng)

    # notebook = False # pop-up window
    # cpos = None

    # notebook = True  # inline
    # cpos = \
    # [(410.42840368821584, -176.08643814594834, 230.59557647292922),
    #  (4.909560043666214, -1.6530085484822958, -26.67373480252808),
    #  (-0.4692582625497324, 0.18316710716520687, 0.8638555978180245)]

    notebook = True  # inline
    cpos = None

    same_color_bar = False

    plot_graph_multi_3d_from_G_networkx_list(
            G_gen_list[:ng], 
            out_name=out_name, 
            nr=None,
            attr=attr,
            attr_label_list=attr_label_list, 
            attr_cmap_list=attr_cmap_list,
            title_list=None, title_fontsize=12,
            notebook=notebook, window_size=window_size, save_fig_png=save_fig_png, off_screen=off_screen,
            filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
            with_labels=False, same_color_bar=same_color_bar, show_color_bar=True,
            kwargs_edges=kwargs_edges, kwargs_pts=kwargs_pts, kwargs_scalar_bar=kwargs_scalar_bar, kwargs_pts_labels=kwargs_pts_labels,
            cpos=cpos, print_cpos=False)


# In[ ]:


# Select single graph
# ===================
ind = 0
G = G_gen_list[ind]

out_name = f'gen_graphs_real_{ind}'


# In[ ]:


print('Plot single selected generated graph (2D)...')

# Plot 2D - single generated graph
# =======
kwds = kwds_single.copy()

figsize = figsize_single
# -----

plot_graph_single_2d_from_G_networkx(
        G, 
        out_name=out_name, 
        attr=attr,
        attr_label_list=attr_label_list, 
        attr_cmap_list=attr_cmap_list,
        title=None, title_fontsize=12,
        figsize=figsize, save_fig_png=save_fig_png, 
        filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
        with_labels=False, show_color_bar=True,
        show=plt_show,
        **kwds)


# In[ ]:


# %%skip_if dim == 2
if dim == 3:
    print('Plot single selected generated graph (3D)...')

    # Plot 3D - single graph
    # =======
    kwargs_edges = kwargs_edges_single.copy()
    kwargs_pts = kwargs_pts_single.copy()
    kwargs_pts_labels = kwargs_pts_labels_single.copy()
    kwargs_scalar_bar = kwargs_scalar_bar_single.copy()

    window_size = window_size_single
    # -----

    # notebook = False # pop-up window
    # cpos = None

    # notebook = True  # inline
    # cpos = \
    # [(517786.43175783526, 171033.62200323722, 3228.2267989145184),
    #  (516160.734375, 171889.74, 813.3000061035157),
    #  (-0.798043530972574, 0.13872899139143935, 0.5864134971334989)]

    notebook = True  # inline
    cpos = None

    plot_graph_single_3d_from_G_networkx(
            G, 
            out_name=out_name, 
            attr=attr,
            attr_label_list=attr_label_list, 
            attr_cmap_list=attr_cmap_list,
            title=None, title_fontsize=12,
            notebook=notebook, window_size=window_size, save_fig_png=save_fig_png, off_screen=off_screen,
            filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
            with_labels=False, show_color_bar=True,
            kwargs_edges=kwargs_edges, kwargs_pts=kwargs_pts, kwargs_scalar_bar=kwargs_scalar_bar, kwargs_pts_labels=kwargs_pts_labels,
            cpos=cpos, print_cpos=False)


# In[ ]:


print('Plot histogram of node features...')

# Histogram of each node feature (coordinates of postion and attributes)
# ======================================================================
if dim==2:
    leg = ['x-coord', 'y-coord']
    col = ['tab:blue', 'tab:orange']
else:
    leg = ['x-coord', 'y-coord', 'z-coord']
    col = ['tab:blue', 'tab:orange', 'tab:purple']

if attr is not None:
    leg = leg + attr_label_list
    col = col + attr_ncomp*['tab:green']

ng = len(leg)
nr = int(np.sqrt(ng))
nc = ng//nr + (ng%nr>0)

figsize = (figsize_lh3[0], figsize_lh3[1]*nr*.75)

out_name = f'gen_graphs_stats_node_features'

x_features = np.vstack([list(networkx.get_node_attributes(G, 'pos').values()) for G in G_gen_list])
if attr is not None:
    v = np.vstack([list(networkx.get_node_attributes(G, attr).values()) for G in G_gen_list])
    x_features = np.hstack((x_features, v))

# Plot
# ----
plt.subplots(nr, nc, figsize=figsize)
for i, (label, color) in enumerate(zip(leg, col)):
    plt.subplot(nr, nc, i+1)
    plt.hist(x_features[:, i], density=True, bins=50, color=color, label=label)
    plt.title(f'{label}')
    # plt.legend()

for i in range(ng, nr*nc):
    plt.subplot(nr, nc, i+1)
    plt.axis('off')

plt.suptitle(f'{out_name} - {len(G_gen_list)} graphs')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}.png')

if plt_show:
    plt.show()
else:
    plt.close()


# In[ ]:


if save_fig_png:
    fig_counter = fig_counter+1


# ## Save / Export

# ### Save the list of generated graphs

# In[ ]:


print('Save / export the list of generated graphs...')

# Save pickle
with open(filename_gen_graph_pk, 'wb') as f: pickle.dump(G_gen_list, file=f)

# Save txt
if attr is not None:
    node_attrs      = ['pos', attr]
else:
    node_attrs      = ['pos']

save_networkx_graph_list(
    G_gen_list,
    out_dir, 
    filename_gen_graph_basename_txt,
    suffix_nodes='_nodes.dat',
    suffix_edges='_links.dat', 
    delimiter_nodes=' ',
    delimiter_edges=' ',
    node_attrs=node_attrs,
    fmt_nodes='%.10g',
    fmt_edges='%i')

