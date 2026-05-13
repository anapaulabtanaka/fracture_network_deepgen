# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: GraphGEN
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model for graph generation (topology) - playing
#

# %%
import networkx
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# import pyvista as pv
import os

import pickle
import dill

# %%
# Choose backend for matplotlib
# -----------------------------
from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('matplotlib', 'inline')

# Or simply:
# # %matplotlib widget
# # %matplotlib inline

# %%
# # Choose backend for pyvista with jupyter
# # ---------------------------------------
# # pv.set_jupyter_backend('trame')  # 3D-interactive plots
# pv.set_jupyter_backend('static') # static plots

# # Notes:
# # -> ignored if run in a standard python shell
# # -> use keyword argument "notebook=False" in Plotter() to open figure in a pop-up window

# %% [markdown]
# ## Load local functions 

# %%
print('Load local functions...')

# import sys
# sys.path.insert(1, '../utils/')

# from graph_utils import *
# from graph_rnn import *
# from ml_utils import *
 
with open('../utils/general_utils.py') as f: exec(f.read())
with open('../utils/graph_utils.py') as f: exec(f.read())
with open('../utils/graph_rnn.py') as f: exec(f.read())
with open('../utils/ml_utils.py') as f: exec(f.read())

# %% [markdown]
# ## Load parameters
#
# Some parameters (dimension / attribute considered and indexes / parameters for plotting graphs)
#

# %%
print('Load parameters...')

# from params import *

with open('params.py') as f: exec(f.read())

# %% [markdown]
# ## Output settings

# %%
print('Define output settings...')

# Output directory (for saving)
# -----------------------------
fig_dir = 'fig'      # PARAMS

plt_show = True      # PARAMS (show graphics 2D ?)
# off_screen = False   # PARAMS (show graphics 3D ?)

save_fig_png = True  # PARAMS
fig_prefix = '02'    # PARAMS

fig_counter = 0

if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)


# %% [markdown]
# ## Input settings
# For loading data set and model (trained).
#
# *Note:* corresponds to "Output settings" in `*_train.ipynb`.

# %%
print('Define input settings...')

# Input directory (for loading)
# -----------------------------
in_dir = 'out_graphRNN_model' # input directory

if not os.path.isdir(in_dir):
    print('ERROR: no input directory')

# Files for loading data set (pickle) (see further)
# -------------------------------------------------
filename_data_set = os.path.join(in_dir, f'data_set.pickle')

# Files for loading networks (rnn_G and rnn_E) (see further)
# ----------------------------------------------------------
filename_hyper_param_G = os.path.join(in_dir, 'rnn_G_hyper_params.txt')
filename_hyper_param_E = os.path.join(in_dir, 'rnn_E_hyper_params.txt')
filename_param_G = os.path.join(in_dir, 'rnn_G.params')
filename_param_E = os.path.join(in_dir, 'rnn_E.params')


# %% [markdown]
# ## Data set

# %% [markdown]
# ### Load the data set

# %%
print('Load the data set (graphRNN)...')

# Load data set
with open(filename_data_set, 'rb') as f: data_set = dill.load(f)

# %% [markdown]
# ### Show first (distinct) graphs of the data set list

# %%
print('Plot first graphs (topology) of the data set list...')

# Plot first graphs - 2d - topology only
# ======================================
kwds = kwds_multi.copy()

figsize = figsize_multi
# -----

ng = 16

ng = min(len(data_set.G_list), ng)
nr = int(np.sqrt(ng))
nc = ng//nr + (ng%nr>0)

# Plot
# ----
plt.subplots(nr, nc, figsize=figsize)
for i, G in enumerate(data_set.G_list[:ng]):
    plt.subplot(nr, nc, i+1)
    networkx.draw(G, with_labels=False, **kwds)
    plt.title(f'n_nodes={G.number_of_nodes()}')

    # plot_graph_2d(G, attr=None, with_labels=False, **kwds)
    # # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # # plt.axis('on')
    # plt.axis('equal')
    # plt.title(f'Graph #{i}\n(n_nodes={G.number_of_nodes()})')

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


# %% [markdown]
# ### Load data via a data loader, and plot first encoded adjacency matrices

# %%
print('Define data loader...')

# Data loader (pytorch)
# ---------------------
batch_size = 6
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)

# %%
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


# %% [markdown]
# ## RNN model for graph generation

# %% [markdown]
# ### Load the model (hyper parameters and parameters)

# %%
print('Load the model (hyper parameters and parameters) (graphRNN)...')

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

# %% [markdown]
# ### Display the model design

# %%
print('Display the model (graphRNN)...')

print('\n')
print('rnn_G\n-----')
print(rnn_G)
print(f'Number of (learnable) params: {nb_net_params(rnn_G)}')

print('\n')
print('rnn_E\n-----')
print(rnn_E)
print(f'Number of (learnable) params: {nb_net_params(rnn_E)}')

# %% [markdown]
# ### Display the model parameters

# %%
# rnn_G.state_dict() # display parameters
# rnn_E.state_dict() # display parameters

# %% [markdown]
# ## Generate graphs

# %% [markdown]
# ### Generate several graphs

# %%
# print('Generate graphs...')

# n_graph = 100
# max_n_nodes = 10000 # should not be reached...

# torch.random.manual_seed(2304)

# t1 = time.time()
# G_gen_list = generate_graph(
#     rnn_G,
#     rnn_E,
#     max_n_nodes=max_n_nodes,
#     n_graph=n_graph,
#     force_node1=True,
#     return_encoded=False,
#     device=default_device()
# )
# t2 = time.time()
# print(f'Elapsed time for generating {n_graph} graph(s): {t2-t1:.3g} s')

# %%
print('Generate graphs...')

n_graph = 100
max_n_nodes = 10000 # should not be reached...

min_n_nodes = 5 # will re-draw graph(s) if fewer nodes

torch.random.manual_seed(2304)

t1 = time.time()
threshold = 0.0  # 0.0 = original Bernoulli behavior (no threshold); increase only after retraining with threshold

G_gen_list = generate_graph_min_n_nodes(
    rnn_G,
    rnn_E,
    min_n_nodes=min_n_nodes,
    max_n_nodes=max_n_nodes,
    n_graph=n_graph,
    force_node1=True,
    return_encoded=False,
    threshold=threshold,
    device=default_device()
)
t2 = time.time()
print(f'Elapsed time for generating {n_graph} graph(s): {t2-t1:.3g} s')

# %% [markdown]
# #### Show first generated graphs

# %%
print('Plot first generated graphs (topology)...')

kwds = kwds_multi.copy()

figsize = figsize_multi
# -----

index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # Index of generated graph to show
ng = len(index)
nr = int(np.sqrt(ng))
nc = ng//nr + (ng%nr>0)

# Plot
# ----
plt.subplots(nr, nc, figsize=figsize)
for i, k in enumerate(index):
    G = G_gen_list[k] # generated graph
    plt.subplot(nr, nc, i+1)
    networkx.draw(G, with_labels=False, **kwds)
    plt.title(f'n_nodes={G.number_of_nodes()}')

    # plot_graph_2d(G, attr=None, with_labels=False, **kwds)
    # # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # # plt.axis('on')
    # plt.axis('equal')
    # plt.title(f'Graph #{i}\n(n_nodes={G.number_of_nodes()})')

for i in range(ng, nr*nc):
    plt.subplot(nr, nc, i+1)
    plt.axis('off')

plt.suptitle(f'graphRNN - generated graphs')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_graphRNN_generated_graphs.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# %%
print('Plot first generated graphs (topology) and [encoded] adjacency matrix...')

# For plotting graphs
kwds = kwds_multi.copy()

figsize = figsize_lh3
# -----

# Plot: graph, adj. matrix, encoded adj. matrix

index = [0, 1, 2, 3] # Index of generated graph

for k in index:
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
    plt.title(f'Generated graph #{k}\n({G.number_of_nodes()} nodes)')

    plt.subplot(1, 3, 2)
    plt.imshow(adj_mat_csr.toarray(), interpolation='none')
    plt.title(f'Adjacency matrix, bw={csr_array_bw(adj_mat_csr)}')

    plt.subplot(1, 3, 3)
    if adj_seq_array.shape[0] > 0:
        plt.imshow(adj_seq_array, origin='upper', extent=[0.5, adj_seq_array.shape[1]+0.5, adj_seq_array.shape[0]+0.5, 0.5], interpolation='none')
    else:
        plt.text(0.5, 0.5, '(single node — no edges)', ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
    # plt.title(f'Encoded adj. matrix (max_prev_node={max_prev_node})(row by row):\n node i (i-th row) is linked to prev. nodes i-j (j-th col) ?')
    plt.title(f'Encoded adj. matrix\nmax_prev_node={max_prev_node}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_gen_graph_check_{k}.png')
        # fig_counter = fig_counter+1
    
    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1


# %%
# # print('Plot specific generated graph (topology) and [encoded] adjacency matrix...')

# # Plot detail of a generated graph with m nodes (if it exists)

# # m = max_n_nodes
# m = 45
# ind = np.where(np.asarray([G.number_of_nodes() for G in G_gen_list]) == m)[0]
# print(f'Index of generated graphs with {m} nodes: ', ind)
# if len(ind):
#     k = ind[0]
#     G = G_gen_list[k] # generated graph

#     # Get adjacency matrix from generated graph
#     adj_mat_csr = networkx.adjacency_matrix(G)
#     # Encode it
#     max_prev_node = rnn_G.input_size
#     adj_seq_array = encode_adj(adj_mat_csr, max_prev_node=max_prev_node)

#     # Plot
#     plt.subplots(1, 3, figsize=figsize)

#     plt.subplot(1, 3, 1)
#     networkx.draw(G, with_labels=False, **kwds)
#     plt.title(f'Generated graph #{k}\n({G.number_of_nodes()} nodes)')

#     plt.subplot(1, 3, 2)
#     plt.imshow(adj_mat_csr.toarray(), interpolation='none')
#     plt.title(f'Adjacency matrix, bw={csr_array_bw(adj_mat_csr)}')

#     plt.subplot(1, 3, 3)
#     plt.imshow(adj_seq_array, origin='upper', extent=[0.5, adj_seq_array.shape[1]+0.5, adj_seq_array.shape[0]+0.5, 0.5], interpolation='none')
#     plt.title(f'Encoded adj. matrix (max_prev_node={max_prev_node})(row by row):\n node i (i-th row) is linked to prev. nodes i-j (j-th col) ?')

#     if save_fig_png:
#         plt.tight_layout()
#         plt.savefig(f'{fig_dir}/{fig_prefix}gen_graph_topo_2d_check_{k}.png')
    
#     if plt_show:
#         plt.show()
#     else:
#         plt.close()


# %% [markdown]
# ## Statistics - number of nodes

# %%
# Colors for further graphs
col_gen = 'tab:blue'
col_data = 'tab:orange'

# %% [markdown]
# ### Number of nodes

# %%
print('Compute and plot statistics - number of nodes...')

figsize = figsize_lh3

# Histogram of number of nodes
n_nodes_gen  = np.asarray([G.number_of_nodes() for G in G_gen_list])
n_nodes_data = np.asarray([G.number_of_nodes() for G in data_set.G_list])
n_nodes_data = n_nodes_data[data_set.G_index_list] # take the rigth number of times the result of each graph in data set

# Plot
vmin = min(n_nodes_gen.min(), n_nodes_data.min()) - 0.5
vmax = max(n_nodes_gen.max(), n_nodes_data.max()) + 0.5
nb = min(int(vmax - vmin), 50)
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(n_nodes_gen,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen. ({len(n_nodes_gen)})')
plt.hist(n_nodes_data, density=True, bins=bins, color=col_data, alpha=.5, label=f'data ({len(n_nodes_data)})')
plt.legend()
plt.title(f'Nb of nodes (max gen.: {n_nodes_gen.max()}, data: {n_nodes_data.max()})')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_nb_nodes.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %% [markdown]
# ## Statistics on graph nodes
# Compute some statistics on nodes of every graph in data set and in generated list (and set results as node attributes).
# The statistics measures computed are: *degree, degree_centrality, closeness_centrality, betweenness_centrality*.
#
# <!--
# The keys and related statistics measures are:
# | key  | stat. measure (on a single graph)|
# |:-----|:---------------------------------|
# | 'n_nodes'                 | number of nodes (int)
# | 'degree'                  | degree of each node (list of ints)
# | 'degree_centrality'       | degree centrality of each node (list of floats)
# | 'closeness_centrality'    | closeness centrality of each node (list of floats)
# | 'betweenness_centrality'  | betweenness centrality (normalized) of each node (list of floats)
# -->

# %%
print('Compute and plot statistics on graph nodes...')

# %%
# Compute statistics on nodes (and set results as node attributes) for each graph in data set
for G in data_set.G_list:
    networkx.set_node_attributes(G, dict(networkx.degree(G)), 'degree')
    networkx.set_node_attributes(G, networkx.degree_centrality(G), 'degree_centrality')
    networkx.set_node_attributes(G, networkx.closeness_centrality(G), 'closeness_centrality')
    networkx.set_node_attributes(G, networkx.betweenness_centrality(G), 'betweenness_centrality')

# %%
# Compute statistics on nodes (and set results as node attributes) for each generated graph
for G in G_gen_list:
    networkx.set_node_attributes(G, dict(networkx.degree(G)), 'degree')
    networkx.set_node_attributes(G, networkx.degree_centrality(G), 'degree_centrality')
    networkx.set_node_attributes(G, networkx.closeness_centrality(G), 'closeness_centrality')
    networkx.set_node_attributes(G, networkx.betweenness_centrality(G), 'betweenness_centrality')

# %% [markdown]
# ### Degree

# %%
# Attribute (key)
attr_stat = 'degree'

dist_gen  = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in G_gen_list]
dist_data = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in data_set.G_list]
dist_data = [dist_data[i] for i in data_set.G_index_list] # take the rigth number of times the result of each graph in data set
# Note: each entry in the lists above is a an array of values of length equal to 
# the number of nodes in the corresponding graph

# %%
figsize = figsize_lh3

# %%
# Histogram over all generated graphs (together) and all graphs in data set (together)
dist_gen_all  = np.hstack(dist_gen)
dist_data_all = np.hstack(dist_data)

# Plot
vmin = min(dist_gen_all.min(), dist_data_all.min()) - 0.01
vmax = max(dist_gen_all.max(), dist_data_all.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_all,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_all, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : over all graphs')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_over_all_graphs.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Histogram of mean values (unique value per graph)
name = 'mean'
dist_gen_u  = np.asarray([d.mean() for d in dist_gen])
dist_data_u = np.asarray([d.mean() for d in dist_data])

# Plot
vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : {name}')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Histogram of std values (unique value per graph)
name = 'std'
dist_gen_u  = np.asarray([d.std() for d in dist_gen])
dist_data_u = np.asarray([d.std() for d in dist_data])

# Plot
vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : {name}')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
figsize = figsize_lh3

# %%
# Box plot for every graph separately
name = 'boxplot_details'
vmin = min(np.asarray([d.min() for d in dist_gen]).min(), np.asarray([d.min() for d in dist_data]).min()) - 0.01
vmax = max(np.asarray([d.max() for d in dist_gen]).max(), np.asarray([d.max() for d in dist_data]).max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

# Plot
plt.subplots(2, 1, figsize=figsize)

plt.subplot(2,1,1)
labels = len(dist_gen)*['']
for i in range(0, len(dist_gen), 5):
    labels[i] = str(i)
bp = plt.boxplot(dist_gen, patch_artist = True, labels=labels)
for patch in bp['boxes']:
    patch.set_facecolor(col_gen)
plt.ylim(ylim)
plt.title(f'{attr_stat}: distribution for each generated graph')

plt.subplot(2,1,2)
labels = len(dist_data)*['']
for i in range(0, len(dist_data), 5):
    labels[i] = str(i)
bp = plt.boxplot(dist_data, patch_artist = True, labels=labels)
for patch in bp['boxes']:
    patch.set_facecolor(col_data)
plt.ylim(ylim)
plt.title(f'{attr_stat}: distribution for each graph in data set')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
   # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Some quantiles for every graph separately
name = 'quantile_details'
quantile = np.array([0, .05, .1, .25, .5, .75, .9, .95, 1.])
label_list = ['min-max', '5-95%', '10-90%', '25-75%']
alpha_list = [.2, .3, .5, .7]

dist_gen_quantile   = np.asarray([np.quantile(d, q=quantile) for d in dist_gen])  # shape (n_graphs, n_quantiles)
dist_data_quantile  = np.asarray([np.quantile(d, q=quantile) for d in dist_data]) # shape (n_graphs, n_quantiles)

vmin = min(dist_gen_quantile.min(), dist_data_quantile.min()) - 0.01
vmax = max(dist_gen_quantile.max(), dist_data_quantile.max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

# Plot
plt.subplots(2, 1, figsize=figsize)

plt.subplot(2,1,1)
plt.plot(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,i], dist_gen_quantile[:,-i-1], color=col_gen, alpha=alpha_list[i], label=label_list[i])

plt.legend(loc='upper right')
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distribution for each generated graph')

plt.subplot(2,1,2)
plt.plot(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,i], dist_data_quantile[:,-i-1], color=col_data, alpha=alpha_list[i], label=label_list[i])

plt.legend(loc='upper right')
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distribution for each graph in data set')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
if save_fig_png:
    fig_counter = fig_counter+1


# %%
# Print some results
for i, d in enumerate(dist_gen[0:5]):
    d_unique, d_counts = np.unique(d, return_counts=True)
    n = len(d)
    print(f'Gen. graph #{i}, nb nodes = {n:5d}')
    for v, c in zip(d_unique, d_counts):
        print(f'   deg = {v:2d},   count: {c:4d},   prop: {c/n:.4f}')


# %% [markdown]
# ### Degree centrality

# %%
# Attribute (key)
attr_stat = 'degree_centrality'

dist_gen  = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in G_gen_list]
dist_data = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in data_set.G_list]
dist_data = [dist_data[i] for i in data_set.G_index_list] # take the rigth number of times the result of each graph in data set
# Note: each entry in the lists above is a an array of values of length equal to 
# the number of nodes in the corresponding graph

# %%
figsize = figsize_lh3

# %%
# Histogram over all generated graphs (together) and all graphs in data set (together)
dist_gen_all  = np.hstack(dist_gen)
dist_data_all = np.hstack(dist_data)

# Plot
vmin = min(dist_gen_all.min(), dist_data_all.min()) - 0.01
vmax = max(dist_gen_all.max(), dist_data_all.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_all,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_all, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : over all graphs')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_over_all_graphs.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Histogram of mean values (unique value per graph)
name = 'mean'
dist_gen_u  = np.asarray([d.mean() for d in dist_gen])
dist_data_u = np.asarray([d.mean() for d in dist_data])

# Plot
vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : {name}')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Histogram of std values (unique value per graph)
name = 'std'
dist_gen_u  = np.asarray([d.std() for d in dist_gen])
dist_data_u = np.asarray([d.std() for d in dist_data])

# Plot
vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : {name}')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
figsize = figsize_lh3

# %%
# Box plot for every graph separately
name = 'boxplot_details'
vmin = min(np.asarray([d.min() for d in dist_gen]).min(), np.asarray([d.min() for d in dist_data]).min()) - 0.01
vmax = max(np.asarray([d.max() for d in dist_gen]).max(), np.asarray([d.max() for d in dist_data]).max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

# Plot
plt.subplots(2, 1, figsize=figsize)

plt.subplot(2,1,1)
labels = len(dist_gen)*['']
for i in range(0, len(dist_gen), 5):
    labels[i] = str(i)
bp = plt.boxplot(dist_gen, patch_artist = True, labels=labels)
for patch in bp['boxes']:
    patch.set_facecolor(col_gen)
plt.ylim(ylim)
plt.title(f'{attr_stat}: distribution for each generated graph')

plt.subplot(2,1,2)
labels = len(dist_data)*['']
for i in range(0, len(dist_data), 5):
    labels[i] = str(i)
bp = plt.boxplot(dist_data, patch_artist = True, labels=labels)
for patch in bp['boxes']:
    patch.set_facecolor(col_data)
plt.ylim(ylim)
plt.title(f'{attr_stat}: distribution for each graph in data set')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Some quantiles for every graph separately
name = 'quantile_details'
quantile = np.array([0, .05, .1, .25, .5, .75, .9, .95, 1.])
label_list = ['min-max', '5-95%', '10-90%', '25-75%']
alpha_list = [.2, .3, .5, .7]

dist_gen_quantile   = np.asarray([np.quantile(d, q=quantile) for d in dist_gen])  # shape (n_graphs, n_quantiles)
dist_data_quantile  = np.asarray([np.quantile(d, q=quantile) for d in dist_data]) # shape (n_graphs, n_quantiles)

vmin = min(dist_gen_quantile.min(), dist_data_quantile.min()) - 0.01
vmax = max(dist_gen_quantile.max(), dist_data_quantile.max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

# Plot
plt.subplots(2, 1, figsize=figsize)

plt.subplot(2,1,1)
plt.plot(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,i], dist_gen_quantile[:,-i-1], color=col_gen, alpha=alpha_list[i], label=label_list[i])

plt.legend(loc='upper right')
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distribution for each generated graph')

plt.subplot(2,1,2)
plt.plot(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,i], dist_data_quantile[:,-i-1], color=col_data, alpha=alpha_list[i], label=label_list[i])

plt.legend(loc='upper right')
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distribution for each graph in data set')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
if save_fig_png:
    fig_counter = fig_counter+1


# %% [markdown]
# ### Closeness centrality

# %%
# Attribute (key)
attr_stat = 'closeness_centrality'

dist_gen  = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in G_gen_list]
dist_data = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in data_set.G_list]
dist_data = [dist_data[i] for i in data_set.G_index_list] # take the rigth number of times the result of each graph in data set
# Note: each entry in the lists above is a an array of values of length equal to 
# the number of nodes in the corresponding graph


# %%
figsize = figsize_lh3

# %%
# Histogram over all generated graphs (together) and all graphs in data set (together)
dist_gen_all  = np.hstack(dist_gen)
dist_data_all = np.hstack(dist_data)

# Plot
vmin = min(dist_gen_all.min(), dist_data_all.min()) - 0.01
vmax = max(dist_gen_all.max(), dist_data_all.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_all,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_all, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : over all graphs')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_over_all_graphs.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Histogram of mean values (unique value per graph)
name = 'mean'
dist_gen_u  = np.asarray([d.mean() for d in dist_gen])
dist_data_u = np.asarray([d.mean() for d in dist_data])

# Plot
vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : {name}')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Histogram of std values (unique value per graph)
name = 'std'
dist_gen_u  = np.asarray([d.std() for d in dist_gen])
dist_data_u = np.asarray([d.std() for d in dist_data])

# Plot
vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : {name}')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
figsize = figsize_lh3

# %%
# Box plot for every graph separately
name = 'boxplot_details'
vmin = min(np.asarray([d.min() for d in dist_gen]).min(), np.asarray([d.min() for d in dist_data]).min()) - 0.01
vmax = max(np.asarray([d.max() for d in dist_gen]).max(), np.asarray([d.max() for d in dist_data]).max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

# Plot
plt.subplots(2, 1, figsize=figsize)

plt.subplot(2,1,1)
labels = len(dist_gen)*['']
for i in range(0, len(dist_gen), 5):
    labels[i] = str(i)
bp = plt.boxplot(dist_gen, patch_artist = True, labels=labels)
for patch in bp['boxes']:
    patch.set_facecolor(col_gen)
plt.ylim(ylim)
plt.title(f'{attr_stat}: distribution for each generated graph')

plt.subplot(2,1,2)
labels = len(dist_data)*['']
for i in range(0, len(dist_data), 5):
    labels[i] = str(i)
bp = plt.boxplot(dist_data, patch_artist = True, labels=labels)
for patch in bp['boxes']:
    patch.set_facecolor(col_data)
plt.ylim(ylim)
plt.title(f'{attr_stat}: distribution for each graph in data set')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Some quantiles for every graph separately
name = 'quantile_details'
quantile = np.array([0, .05, .1, .25, .5, .75, .9, .95, 1.])
label_list = ['min-max', '5-95%', '10-90%', '25-75%']
alpha_list = [.2, .3, .5, .7]

dist_gen_quantile   = np.asarray([np.quantile(d, q=quantile) for d in dist_gen])  # shape (n_graphs, n_quantiles)
dist_data_quantile  = np.asarray([np.quantile(d, q=quantile) for d in dist_data]) # shape (n_graphs, n_quantiles)

vmin = min(dist_gen_quantile.min(), dist_data_quantile.min()) - 0.01
vmax = max(dist_gen_quantile.max(), dist_data_quantile.max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

# Plot
plt.subplots(2, 1, figsize=figsize)

plt.subplot(2,1,1)
plt.plot(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,i], dist_gen_quantile[:,-i-1], color=col_gen, alpha=alpha_list[i], label=label_list[i])

plt.legend(loc='upper right')
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distribution for each generated graph')

plt.subplot(2,1,2)
plt.plot(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,i], dist_data_quantile[:,-i-1], color=col_data, alpha=alpha_list[i], label=label_list[i])

plt.legend(loc='upper right')
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distribution for each graph in data set')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# %%
if save_fig_png:
    fig_counter = fig_counter+1


# %% [markdown]
# ### Betweenness centrality

# %%
# Attribute (key)
attr_stat = 'betweenness_centrality'

dist_gen  = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in G_gen_list]
dist_data = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in data_set.G_list]
dist_data = [dist_data[i] for i in data_set.G_index_list] # take the rigth number of times the result of each graph in data set
# Note: each entry in the lists above is a an array of values of length equal to 
# the number of nodes in the corresponding graph


# %%
figsize = figsize_lh3

# %%
# Histogram over all generated graphs (together) and all graphs in data set (together)
dist_gen_all  = np.hstack(dist_gen)
dist_data_all = np.hstack(dist_data)

# Plot
vmin = min(dist_gen_all.min(), dist_data_all.min()) - 0.01
vmax = max(dist_gen_all.max(), dist_data_all.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_all,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_all, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : over all graphs')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_over_all_graphs.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Histogram of mean values (unique value per graph)
name = 'mean'
dist_gen_u  = np.asarray([d.mean() for d in dist_gen])
dist_data_u = np.asarray([d.mean() for d in dist_data])

# Plot
vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : {name}')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Histogram of std values (unique value per graph)
name = 'std'
dist_gen_u  = np.asarray([d.std() for d in dist_gen])
dist_data_u = np.asarray([d.std() for d in dist_data])

# Plot
vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.legend()
plt.title(f'{attr_stat} : {name}')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
figsize = figsize_lh3

# %%
# Box plot for every graph separately
name = 'boxplot_details'
vmin = min(np.asarray([d.min() for d in dist_gen]).min(), np.asarray([d.min() for d in dist_data]).min()) - 0.01
vmax = max(np.asarray([d.max() for d in dist_gen]).max(), np.asarray([d.max() for d in dist_data]).max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

# Plot
plt.subplots(2, 1, figsize=figsize)

plt.subplot(2,1,1)
labels = len(dist_gen)*['']
for i in range(0, len(dist_gen), 5):
    labels[i] = str(i)
bp = plt.boxplot(dist_gen, patch_artist = True, labels=labels)
for patch in bp['boxes']:
    patch.set_facecolor(col_gen)
plt.ylim(ylim)
plt.title(f'{attr_stat}: distribution for each generated graph')

plt.subplot(2,1,2)
labels = len(dist_data)*['']
for i in range(0, len(dist_data), 5):
    labels[i] = str(i)
bp = plt.boxplot(dist_data, patch_artist = True, labels=labels)
for patch in bp['boxes']:
    patch.set_facecolor(col_data)
plt.ylim(ylim)
plt.title(f'{attr_stat}: distribution for each graph in data set')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# Some quantiles for every graph separately
name = 'quantile_details'
quantile = np.array([0, .05, .1, .25, .5, .75, .9, .95, 1.])
label_list = ['min-max', '5-95%', '10-90%', '25-75%']
alpha_list = [.2, .3, .5, .7]

dist_gen_quantile   = np.asarray([np.quantile(d, q=quantile) for d in dist_gen])  # shape (n_graphs, n_quantiles)
dist_data_quantile  = np.asarray([np.quantile(d, q=quantile) for d in dist_data]) # shape (n_graphs, n_quantiles)

vmin = min(dist_gen_quantile.min(), dist_data_quantile.min()) - 0.01
vmax = max(dist_gen_quantile.max(), dist_data_quantile.max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

# Plot
plt.subplots(2, 1, figsize=figsize)

plt.subplot(2,1,1)
plt.plot(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,i], dist_gen_quantile[:,-i-1], color=col_gen, alpha=alpha_list[i], label=label_list[i])

plt.legend(loc='upper right')
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distribution for each generated graph')

plt.subplot(2,1,2)
plt.plot(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,i], dist_data_quantile[:,-i-1], color=col_data, alpha=alpha_list[i], label=label_list[i])

plt.legend(loc='upper right')
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distribution for each graph in data set')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_{name}.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
if save_fig_png:
    fig_counter = fig_counter+1


# %% [markdown]
# ### Several statistics in one figure

# %%
print('Plot statistics summary...')

figsize = figsize_equal_l6

# Histogram of number of nodes
n_nodes_gen  = np.asarray([G.number_of_nodes() for G in G_gen_list])
n_nodes_data = np.asarray([G.number_of_nodes() for G in data_set.G_list])
n_nodes_data = n_nodes_data[data_set.G_index_list] # take the rigth number of times the result of each graph in data set

attr_stat_list = ['degree', 'degree_centrality', 'closeness_centrality', 'betweenness_centrality']

fig, ax = plt.subplots(5, 5, figsize=figsize)

# ------ col 0 -------#
for ir in range(5):
    # ------ row 0 -------#
    plt.sca(ax[0, 0])
    title = 'over all graphs'
    plt.text(.5, .5, title, horizontalalignment='center', verticalalignment='center')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('off')

    # ------ row 1 -------#
    plt.sca(ax[1, 0])
    title = 'mean'
    plt.text(.5, .5, title, horizontalalignment='center', verticalalignment='center')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('off')
    
    # ------ row 2 -------#
    plt.sca(ax[2, 0])
    title = 'std'
    plt.text(.5, .5, title, horizontalalignment='center', verticalalignment='center')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('off')
    
    # ------ row 3 -------#
    plt.sca(ax[3, 0])
    title = 'quantile - gen'
    # title = 'box plot - gen'
    plt.text(.5, .5, title, horizontalalignment='center', verticalalignment='center')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('off')
    
    # ------ row 4 -------#
    plt.sca(ax[4, 0])
    title = 'quantile - data set'
    # title = 'box plot - data set'
    plt.text(.5, .5, title, horizontalalignment='center', verticalalignment='center')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.axis('off')

# ------ col > 0 -------#
for ic, attr_stat in enumerate(attr_stat_list):
    # Get distribution of values for each graph (in both set)
    dist_gen  = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in G_gen_list]
    dist_data = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in data_set.G_list]
    dist_data = [dist_data[i] for i in data_set.G_index_list] # take the rigth number of times the result of each graph in data set

    # ------ row 0 -------#
    plt.sca(ax[0, ic+1])

    # Histogram over all generated graphs (together) and all graphs in data set (together)
    dist_gen_all  = np.hstack(dist_gen)
    dist_data_all = np.hstack(dist_data)
    
    vmin = min(dist_gen_all.min(), dist_data_all.min()) - 0.01
    vmax = max(dist_gen_all.max(), dist_data_all.max()) + 0.01
    nb = 50
    bins = np.linspace(vmin, vmax, nb+1)

    plt.hist(dist_gen_all,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
    plt.hist(dist_data_all, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
    plt.legend()
    plt.title(f'{attr_stat}')

    # ------ row 1 -------#
    plt.sca(ax[1, ic+1])

    # Histogram of mean values (unique value per graph)
    name = 'mean'
    dist_gen_u  = np.asarray([d.mean() for d in dist_gen])
    dist_data_u = np.asarray([d.mean() for d in dist_data])

    vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
    vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
    nb = 50
    bins = np.linspace(vmin, vmax, nb+1)

    plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
    plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
    # plt.legend()
    # plt.title(f'{attr_stat} : {name}')
    #plt.grid()

    # ------ row 2 -------#
    plt.sca(ax[2, ic+1])

    # Histogram of std values (unique value per graph)
    name = 'std'
    dist_gen_u  = np.asarray([d.std() for d in dist_gen])
    dist_data_u = np.asarray([d.std() for d in dist_data])

    vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
    vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
    nb = 50
    bins = np.linspace(vmin, vmax, nb+1)

    plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
    plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
    # plt.legend()
    # plt.title(f'{attr_stat} : {name}')
    #plt.grid()

    # ------ row 3 and 4 -------#
    name = 'quantile'
    quantile = np.array([0, .05, .1, .25, .5, .75, .9, .95, 1.])
    label_list = ['min-max', '5-95%', '10-90%', '25-75%']
    alpha_list = [.2, .3, .5, .7]

    dist_gen_quantile   = np.asarray([np.quantile(d, q=quantile) for d in dist_gen])  # shape (n_graphs, n_quantiles)
    dist_data_quantile  = np.asarray([np.quantile(d, q=quantile) for d in dist_data]) # shape (n_graphs, n_quantiles)

    vmin = min(dist_gen_quantile.min(), dist_data_quantile.min()) - 0.01
    vmax = max(dist_gen_quantile.max(), dist_data_quantile.max()) + 0.01
    ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

    # ------ row 3 -------#
    plt.sca(ax[3, ic+1])

    # Some quantiles for every graph separately - generated graph
    plt.plot(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,4], lw=.5, c='black', label=f'50%')
    for i in range(4):
        plt.fill_between(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,i], dist_gen_quantile[:,-i-1], color=col_gen, alpha=alpha_list[i], label=label_list[i])

    if ic == 0:
        plt.legend()
    plt.ylim(ylim)
    plt.grid()
    # plt.title(f'{attr_stat}: distribution for each generated graph')

    # ------ row 4 -------#
    plt.sca(ax[4, ic+1])

    # Some quantiles for every graph separately - data set
    plt.plot(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,4], lw=.5, c='black', label=f'50%')
    for i in range(4):
        plt.fill_between(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,i], dist_data_quantile[:,-i-1], color=col_data, alpha=alpha_list[i], label=label_list[i])

    if ic == 0:
        plt.legend()
    plt.ylim(ylim)
    plt.grid()
    # plt.title(f'{attr_stat}: distribution for each graph in data set')

    # # ------ row 3 and 4 -------#
    # vmin = min(np.asarray([d.min() for d in dist_gen]).min(), np.asarray([d.min() for d in dist_data]).min()) - 0.01
    # vmax = max(np.asarray([d.max() for d in dist_gen]).max(), np.asarray([d.max() for d in dist_data]).max()) + 0.01
    # ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

    # # ------ row 3 -------#
    # plt.sca(ax[3, ic])

    # # Box plot for every graph separately - generated graph
    # labels = len(dist_gen)*['']
    # for i in range(0, len(dist_gen), 10):
    #     labels[i] = str(i)
    # bp = plt.boxplot(dist_gen, patch_artist = True, labels=labels)
    # for patch in bp['boxes']:
    #     patch.set_facecolor(col_gen)
    # plt.ylim(ylim)
    # # plt.title(f'{attr_stat}: distribution for each generated graph')

    # # ------ row 4 -------#
    # plt.sca(ax[4, ic])

    # # Box plot for every graph separately - data set
    # labels = len(dist_data)*['']
    # for i in range(0, len(dist_data), 10):
    #     labels[i] = str(i)
    # bp = plt.boxplot(dist_data, patch_artist = True, labels=labels)
    # for patch in bp['boxes']:
    #     patch.set_facecolor(col_data)
    # plt.ylim(ylim)
    # # plt.title(f'{attr_stat}: distribution for each graph in data set')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_summary.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# %%
# # Histogram of number of nodes
# n_nodes_gen  = np.asarray([G.number_of_nodes() for G in G_gen_list])
# n_nodes_data = np.asarray([G.number_of_nodes() for G in data_set.G_list])
# n_nodes_data = n_nodes_data[data_set.G_index_list] # take the rigth number of times the result of each graph in data set

# attr_stat_list = ['degree', 'degree_centrality', 'closeness_centrality', 'betweenness_centrality']

# fig, ax = plt.subplots(5, 4, figsize=(12, 15))
# for ic, attr_stat in enumerate(attr_stat_list):
#     # Get distribution of values for each graph (in both set)
#     dist_gen  = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in G_gen_list]
#     dist_data = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in data_set.G_list]
#     dist_data = [dist_data[i] for i in data_set.G_index_list] # take the rigth number of times the result of each graph in data set

#     # ------ row 0 -------#
#     plt.sca(ax[0, ic])

#     # Histogram over all generated graphs (together) and all graphs in data set (together)
#     dist_gen_all  = np.hstack(dist_gen)
#     dist_data_all = np.hstack(dist_data)
    
#     vmin = min(dist_gen_all.min(), dist_data_all.min()) - 0.01
#     vmax = max(dist_gen_all.max(), dist_data_all.max()) + 0.01
#     nb = 50
#     bins = np.linspace(vmin, vmax, nb+1)

#     plt.hist(dist_gen_all,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
#     plt.hist(dist_data_all, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
#     plt.legend()
#     plt.title(f'{attr_stat}\nover all graphs')

#     # ------ row 1 -------#
#     plt.sca(ax[1, ic])

#     # Histogram of mean values (unique value per graph)
#     name = 'mean'
#     dist_gen_u  = np.asarray([d.mean() for d in dist_gen])
#     dist_data_u = np.asarray([d.mean() for d in dist_data])

#     vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
#     vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
#     nb = 50
#     bins = np.linspace(vmin, vmax, nb+1)

#     plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
#     plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
#     # plt.legend()
#     # plt.title(f'{attr_stat} : {name}')
#     plt.title(f'{name}')
#     #plt.grid()

#     # ------ row 2 -------#
#     plt.sca(ax[2, ic])

#     # Histogram of std values (unique value per graph)
#     name = 'std'
#     dist_gen_u  = np.asarray([d.std() for d in dist_gen])
#     dist_data_u = np.asarray([d.std() for d in dist_data])

#     vmin = min(dist_gen_u.min(), dist_data_u.min()) - 0.01
#     vmax = max(dist_gen_u.max(), dist_data_u.max()) + 0.01
#     nb = 50
#     bins = np.linspace(vmin, vmax, nb+1)

#     plt.hist(dist_gen_u,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
#     plt.hist(dist_data_u, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
#     # plt.legend()
#     # plt.title(f'{attr_stat} : {name}')
#     plt.title(f'{name}')
#     #plt.grid()

#     # ------ row 3 and 4 -------#
#     name = 'quantile'
#     quantile = np.array([0, .05, .1, .25, .5, .75, .9, .95, 1.])
#     label_list = ['min-max', '5-95%', '10-90%', '25-75%']
#     alpha_list = [.2, .3, .5, .7]

#     dist_gen_quantile   = np.asarray([np.quantile(d, q=quantile) for d in dist_gen])  # shape (n_graphs, n_quantiles)
#     dist_data_quantile  = np.asarray([np.quantile(d, q=quantile) for d in dist_data]) # shape (n_graphs, n_quantiles)

#     vmin = min(dist_gen_quantile.min(), dist_data_quantile.min()) - 0.01
#     vmax = max(dist_gen_quantile.max(), dist_data_quantile.max()) + 0.01
#     ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

#     # ------ row 3 -------#
#     plt.sca(ax[3, ic])

#     # Some quantiles for every graph separately - generated graph
#     plt.plot(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,4], lw=.5, c='black', label=f'50%')
#     for i in range(4):
#         plt.fill_between(np.arange(dist_gen_quantile.shape[0]), dist_gen_quantile[:,i], dist_gen_quantile[:,-i-1], color=col_gen, alpha=alpha_list[i], label=label_list[i])

#     if ic == 0:
#         plt.legend()
#     plt.ylim(ylim)
#     plt.grid()
#     # plt.title(f'{attr_stat}: distribution for each generated graph')
#     plt.title(f'gen. graphs')

#     # ------ row 4 -------#
#     plt.sca(ax[4, ic])

#     # Some quantiles for every graph separately - data set
#     plt.plot(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,4], lw=.5, c='black', label=f'50%')
#     for i in range(4):
#         plt.fill_between(np.arange(dist_data_quantile.shape[0]), dist_data_quantile[:,i], dist_data_quantile[:,-i-1], color=col_data, alpha=alpha_list[i], label=label_list[i])

#     if ic == 0:
#         plt.legend()
#     plt.ylim(ylim)
#     plt.grid()
#     # plt.title(f'{attr_stat}: distribution for each graph in data set')
#     plt.title(f'data set')

#     # # ------ row 3 and 4 -------#
#     # vmin = min(np.asarray([d.min() for d in dist_gen]).min(), np.asarray([d.min() for d in dist_data]).min()) - 0.01
#     # vmax = max(np.asarray([d.max() for d in dist_gen]).max(), np.asarray([d.max() for d in dist_data]).max()) + 0.01
#     # ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

#     # # ------ row 3 -------#
#     # plt.sca(ax[3, ic])

#     # # Box plot for every graph separately - generated graph
#     # labels = len(dist_gen)*['']
#     # for i in range(0, len(dist_gen), 10):
#     #     labels[i] = str(i)
#     # bp = plt.boxplot(dist_gen, patch_artist = True, labels=labels)
#     # for patch in bp['boxes']:
#     #     patch.set_facecolor(col_gen)
#     # plt.ylim(ylim)
#     # # plt.title(f'{attr_stat}: distribution for each generated graph')
#     # plt.title(f'gen. graphs')

#     # # ------ row 4 -------#
#     # plt.sca(ax[4, ic])

#     # # Box plot for every graph separately - data set
#     # labels = len(dist_data)*['']
#     # for i in range(0, len(dist_data), 10):
#     #     labels[i] = str(i)
#     # bp = plt.boxplot(dist_data, patch_artist = True, labels=labels)
#     # for patch in bp['boxes']:
#     #     patch.set_facecolor(col_data)
#     # plt.ylim(ylim)
#     # # plt.title(f'{attr_stat}: distribution for each graph in data set')
#     # plt.title(f'data set')

# if plt_show:
#     plt.show()
# else:
#     plt.close()

