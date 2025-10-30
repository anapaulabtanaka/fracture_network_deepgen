#!/usr/bin/env python
# coding: utf-8

# # Statistics on graphs

# In[1]:


import networkx
import torch, torch_geometric
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

import pyvista as pv
import os

import karstnet as kn

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
# # from graph_rnn import *
# from graph_ddpm import *
# # from ml_utils import *
# from general_utils import *
# from graph_plot import *
# # from magic_utils import *
 
with open('../utils/graph_utils.py') as f: exec(f.read())
# with open('../utils/graph_rnn.py') as f: exec(f.read())
with open('../utils/graph_ddpm.py') as f: exec(f.read())
# with open('../utils/ml_utils.py') as f: exec(f.read())
with open('../utils/general_utils.py') as f: exec(f.read())
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

# In[ ]:


print('Define output settings...')

# Output directory (for saving)
# -----------------------------
fig_dir = 'fig'      # PARAMS

plt_show = False     # PARAMS (show graphics 2D ?)
off_screen = True    # PARAMS (show graphics 3D ?)

save_fig_png = True  # PARAMS
fig_prefix = '07'    # PARAMS

fig_counter = 0

if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)


# ## Load graphs

# ### Load graphs from data set

# In[ ]:


print('Load graphs from data set...')

# Load from pickle file
data_dir = 'data_gen'
filename_graph_collection_data_set = os.path.join(data_dir, f'graph_collection_data_set.pickle')

with open(filename_graph_collection_data_set, 'rb') as f: data_set_G_list = pickle.load(f)


# In[ ]:


# print('Load graphs from data set...')

# # Alternative - read from text files (provides the same lists)
# # ============================================================

# data_dir = 'data_gen'
# filename_base = 'graph_collection_data_set'

# if attr is not None:
#     node_attrs      = ['pos', attr]
#     node_attrs_ind  = [tuple(range(dim)), [dim + i for i in range(attr_ncomp)]]
#     node_attrs_type = ['float', 'float']
# else:
#     node_attrs      = ['pos']
#     node_attrs_ind  = [tuple(range(dim))]
#     node_attrs_type = ['float']

# # Load graph list for data set
# data_set_G_list_2 = load_networkx_graph_list(
#     data_dir, filename_base, 
#     suffix_nodes='_nodes.dat', 
#     suffix_edges='_links.dat', 
#     delimiter_nodes=' ',  
#     delimiter_edges=' ',
#     node_attrs=node_attrs,
#     node_attrs_ind=node_attrs_ind,
#     nodet_attrs_type=node_attrs_type,
#     start_id_at_0=True)

# # Check
# # -----
# ok_list = []
# for i, (G1, G2) in enumerate(zip(data_set_G_list, data_set_G_list_2)):
#     ok = True
#     # print(f'graph #{i} ...')
#     check_list = compare_graph(G1, G2, node_attrs, verbose=0)
#     # print(f'graph #{i:3d} - check_list = {check_list}')
#     ok_list.append(np.all(check_list))

# print(f'`data_set_G_list` and `data_set_G_list_2` are equal ? {np.all(ok_list)}')
# # print(ok_list)


# In[ ]:


# print('Load graphs from data set...')

# # Alternative - read from ddpm data set (generated in notebook 03_graphDDPM_model_train.ipynb) and transform 
# # (provides the same lists)
# # ===========================================================================================================

# in_dir_ddpm = 'out_graphDDPM_model'
# # filename_hyper_param_ddpm_net = os.path.join(in_dir_ddpm, 'ddpm_net_hyper_params.txt')
# # filename_hyper_param_ddpm     = os.path.join(in_dir_ddpm, 'ddpm_hyper_params.txt')
# # filename_param_ddpm     = os.path.join(in_dir_ddpm, 'ddpm.params')
# filename_data_set_ddpm_shift = os.path.join(in_dir_ddpm, f'data_set_shift.txt')
# filename_data_set_ddpm_scale_factor = os.path.join(in_dir_ddpm, f'data_set_scale_factor.txt')
# filename_data_set_ddpm = os.path.join(in_dir_ddpm, f'data_set.pickle')

# # Load data set
# with open(filename_data_set_ddpm, 'rb') as f: data_set = pickle.load(f)

# # Load node features shift and scale factor
# with open(filename_data_set_ddpm_shift, 'r') as f: node_features_shift = np.loadtxt(f)
# with open(filename_data_set_ddpm_scale_factor, 'r') as f: node_features_scale_factor = np.loadtxt(f)

# node_features_shift_inv = - node_features_shift
# node_features_scale_factor_inv = 1.0 / node_features_scale_factor

# n_node_features = len(node_features_scale_factor)

# # Get data_set_G_list
# data_set_G_list_2 = []
# for G_geom in data_set.G_geom_list:
#     # G_geom.x = torch.from_numpy(node_features_scale_factor_inv).to(torch.float)*G_geom.x
#     G = torch_geometric.utils.to_networkx(G_geom, to_undirected=True, node_attrs=['x'])
#     G = multiply_graph_node_features(G, 'x', node_features_scale_factor_inv, inplace=True)
#     G = shift_graph_node_features(G, 'x', node_features_shift_inv, inplace=True)

#     v = np.asarray(list(networkx.get_node_attributes(G, 'x').values()))
#     remove_node_attribute(G, 'x')

#     dict_pos = {i:vi[:dim].tolist() for i, vi in enumerate(v)}
#     networkx.set_node_attributes(G, dict_pos, 'pos')

#     if attr is not None:
#         dict_attr = {i:vi[dim:].tolist() for i, vi in enumerate(v)}
#         networkx.set_node_attributes(G, dict_attr, attr)
#     else:
#         dict_attr = None

#     data_set_G_list_2.append(G)

# # Check
# # -----
# if attr is not None:
#     node_attrs      = ['pos', attr]
# else:
#     node_attrs      = ['pos']
    
# ok_list = []
# for i, (G1, G2) in enumerate(zip(data_set_G_list, data_set_G_list_2)):
#     ok = True
#     # print(f'graph #{i} ...')
#     check_list = compare_graph(G1, G2, node_attrs, verbose=0, atol=1e-07)
#     # print(f'graph #{i:3d} - check_list = {check_list}')
#     ok_list.append(np.all(check_list))

# print(f'`data_set_G_list` and `data_set_G_list_2` are equal ? {np.all(ok_list)}')
# # print(ok_list)


# ### Load list of generated graphs

# In[ ]:


print('Load generated graphs...')

in_dir_gen = 'out_gen_graph' 
filename_gen_graph_pk = os.path.join(in_dir_gen, f'gen_graph_list.pickle')

# Load list of generated graph (G_gen_list: of generated graphs in networkx format)
with open(filename_gen_graph_pk, 'rb') as f: G_gen_list = pickle.load(f)


# In[ ]:


# print('Load generated graphs...')

# # Alternative - read from text files (provides the same lists)
# # ============================================================

# in_dir_gen = 'out_gen_graph' 
# filename_gen_graph_basename_txt = 'gen_graph_list'

# if attr is not None:
#     node_attrs      = ['pos', attr]
#     node_attrs_ind  = [tuple(range(dim)), [dim + i for i in range(attr_ncomp)]]
#     node_attrs_type = ['float', 'float']
# else:
#     node_attrs      = ['pos']
#     node_attrs_ind  = [tuple(range(dim))]
#     node_attrs_type = ['float']

# G_gen_list_2 = load_networkx_graph_list(
#     in_dir_gen, 
#     filename_gen_graph_basename_txt,
#     suffix_nodes='_nodes.dat', 
#     suffix_edges='_links.dat', 
#     delimiter_nodes=' ',  
#     delimiter_edges=' ',
#     node_attrs=node_attrs,
#     node_attrs_ind=node_attrs_ind,
#     nodet_attrs_type=node_attrs_type,
#     start_id_at_0=True)

# # Check
# # -----
# ok_list = []
# for i, (G1, G2) in enumerate(zip(G_gen_list, G_gen_list_2)):
#     ok = True
#     # print(f'graph #{i} ...')
#     check_list = compare_graph(G1, G2, node_attrs, verbose=0)
#     # print(f'graph #{i:3d} - check_list = {check_list}')
#     ok_list.append(np.all(check_list))
#     break

# print(f'`G_gen_list` and `G_gen_list_2` are equal ? {np.all(ok_list)}')
# # print(ok_list)


# ### Set in "Karstnet format"

# In[ ]:


print('Set graphs of data set and generated graphs in "Karstnet format"...')

# Karstnet graph with attribute (properties)
if attr is not None:
    data_set_KG_list = [kn.KGraph(G.edges(), networkx.get_node_attributes(G, 'pos'), 
                                  properties=networkx.get_node_attributes(G, attr), verbose=False) for G in data_set_G_list]

    KG_gen_list = [kn.KGraph(G.edges(), networkx.get_node_attributes(G, 'pos'), 
                             properties=networkx.get_node_attributes(G, attr), verbose=False) for G in G_gen_list]
else:
    data_set_KG_list = [kn.KGraph(G.edges(), networkx.get_node_attributes(G, 'pos'), 
                                  properties=None, verbose=False) for G in data_set_G_list]

    KG_gen_list = [kn.KGraph(G.edges(), networkx.get_node_attributes(G, 'pos'), 
                             properties=None, verbose=False) for G in G_gen_list]


# ## Show first graphs

# ### Graphs of data set

# In[ ]:


print('Plot first graphs of data set (2D)...')

G_list = data_set_G_list[:16]
out_name = 'graphs_data_set'

# 2D view
# =======
kwds = kwds_multi.copy()

figsize = figsize_multi
# -----

same_color_bar = False

plot_graph_multi_2d_from_G_networkx_list(
        G_list, 
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
    print('Plot first graphs of data set (3D)...')

    G_list = data_set_G_list[:9]
    out_name = 'graphs_data_set'

    # Plot first graphs - 3d
    # ======================
    kwargs_edges = kwargs_edges_multi.copy()
    kwargs_pts = kwargs_pts_multi.copy()
    kwargs_pts_labels = kwargs_pts_labels_multi.copy()
    kwargs_scalar_bar = kwargs_scalar_bar_multi.copy()

    window_size = window_size_multi
    # -----

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
            G_list, 
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


# In[15]:


if save_fig_png:
    fig_counter = fig_counter+1


# ### Generated graphs (graphRNN + ddpm)

# In[ ]:


print('Plot first generated graphs (2D)...')

G_list = G_gen_list[:16]
out_name = 'generated_graphs'

# 2D view
# =======
kwds = kwds_multi.copy()

figsize = figsize_multi
# -----

same_color_bar = False

plot_graph_multi_2d_from_G_networkx_list(
        G_list, 
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

    G_list = G_gen_list[:9]
    out_name = 'generated_graphs'

    # Plot first graphs - 3d
    # ======================
    kwargs_edges = kwargs_edges_multi.copy()
    kwargs_pts = kwargs_pts_multi.copy()
    kwargs_pts_labels = kwargs_pts_labels_multi.copy()
    kwargs_scalar_bar = kwargs_scalar_bar_multi.copy()

    window_size = window_size_multi
    # -----

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
            G_list, 
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


# In[18]:


if save_fig_png:
    fig_counter = fig_counter+1


# ## Statistics

# In[19]:


# Colors for further graphs
col_gen = 'tab:blue'
col_data = 'tab:orange'

col2_gen = 'darkblue'
col2_data = 'tab:red'


# ### Number of nodes

# In[ ]:


print('Compute and plot statistics - number of nodes...')

figsize = figsize_lh3

# Histogram of number of nodes
data_n_nodes = np.asarray([G.number_of_nodes() for G in data_set_G_list])
gen_n_nodes  = np.asarray([G.number_of_nodes() for G in G_gen_list])

# Plot
vmin = min(data_n_nodes.min(), gen_n_nodes.min()) - 0.5
vmax = max(data_n_nodes.max(), gen_n_nodes.max()) + 0.5
nb = min(int(vmax - vmin), 50)
bins = np.linspace(vmin, vmax, nb+1)

plt.figure(figsize=figsize)
plt.hist(gen_n_nodes,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen. ({len(gen_n_nodes)})')
plt.hist(data_n_nodes, density=True, bins=bins, color=col_data, alpha=.5, label=f'data ({len(data_n_nodes)})')
plt.legend()
plt.title(f'Nb of nodes (max gen.: {gen_n_nodes.max()}, data: {data_n_nodes.max()})')
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_nb_nodes.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


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

data_x_features = np.vstack([list(networkx.get_node_attributes(G, 'pos').values()) for G in data_set_G_list])
if attr is not None:
    v = np.vstack([list(networkx.get_node_attributes(G, attr).values()) for G in data_set_G_list])
    data_x_features = np.hstack((data_x_features, v))

gen_x_features = np.vstack([list(networkx.get_node_attributes(G, 'pos').values()) for G in G_gen_list])
if attr is not None:
    v = np.vstack([list(networkx.get_node_attributes(G, attr).values()) for G in G_gen_list])
    gen_x_features = np.hstack((gen_x_features, v))

x_features_min = np.vstack((data_x_features.min(axis=0), gen_x_features.min(axis=0))).min(axis=0)
x_features_max = np.vstack((data_x_features.max(axis=0), gen_x_features.max(axis=0))).max(axis=0)
x_features_nb = len(leg)*[50]
x_features_bins = [np.linspace(a, b, nb+1) for a, b, nb in zip(x_features_min, x_features_max, x_features_nb)]

# Plot
# ----
plt.subplots(nr, nc, figsize=figsize)
for i, label in enumerate(leg):
    plt.subplot(nr, nc, i+1)
    plt.hist(data_x_features[:, i], density=True, bins=x_features_bins[i], color=col_data, alpha=.5, label=f'data')
    plt.hist(gen_x_features[:, i],  density=True, bins=x_features_bins[i], color=col_gen,  alpha=.5, label=f'gen.')
    plt.title(f'{label}')
    plt.legend()

for i in range(ng, nr*nc):
    plt.subplot(nr, nc, i+1)
    plt.axis('off')

plt.suptitle(f'Densities of features')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_node_features.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# In[ ]:


# %%skip_if attr_ncomp < 2
if attr_ncomp >= 2:
    print('Cross-plot of node features...')

    # Cross plot of pair of attributes
    # ================================

    figsize = figsize_equal_def

    data_attr = np.vstack([list(networkx.get_node_attributes(G, attr).values()) for G in data_set_G_list])
    gen_attr  = np.vstack([list(networkx.get_node_attributes(G, attr).values()) for G in G_gen_list])

    for i_attr in range(attr_ncomp):
        # loop on attributes
        attr_label_i = attr_label_list[i_attr]
        
        for j_attr in range(i_attr+1, attr_ncomp):
            # loop on attributes
            attr_label_j = attr_label_list[j_attr]

            plt.figure(figsize=figsize)
            plt.scatter(data_attr[:, i_attr], data_attr[:, j_attr], c=col_data, alpha=.1, s=20, marker='o', label='data')
            plt.scatter(gen_attr [:, i_attr], gen_attr [:, j_attr], c=col_gen,  alpha=.1, s=20, marker='o', label='gen')
            plt.xlabel(f'{attr_label_i}')
            plt.ylabel(f'{attr_label_j}')
            plt.grid()
            plt.legend()

            plt.title(f'Cross plot attributes {i_attr} vs {j_attr}')

            if save_fig_png:
                plt.tight_layout()
                plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_cross_plot_{attr}_{i_attr}_{j_attr}.png')
                # fig_counter = fig_counter+1

            if plt_show:
                plt.show()
            else:
                plt.close()


# In[ ]:


if save_fig_png:
    fig_counter = fig_counter+1


# ## Statistics on nodes
# Compute some statistics on nodes of every graph in data set and in generated list (and set results as node attributes).
# The statistics measures computed are: *degree, degree_centrality, closeness_centrality, betweenness_centrality* (all normalized, i.e. the maximum is $1$).
# 
# Below, $n$ denotes the number of nodes in the graph, $u$, $v$ are nodes of the graph, $d(v)$ is the degree of $v$, and $d(u,v)$ is the length of the shortest path between node $u$ and $v$ (in number of edges).
# 
# - $degree\_centrality(v) = d(v)/(n-1)$ 
# - $closeness\_centrality(v) = (n-1)/\sum_{u\neq v}d(u, v)$ 
# - $betweenness\_centrality(v) = \frac{2}{(n-1)(n-2)}\sum_{\{s, t\} : s \neq t, s\neq v\neq t} \frac{\vert\{shortest\ path\ btw\ s\ and\ t\ going\ through\ v\}\vert}{\vert\{shortest\ path\ btw\ s\ and\ t\}\vert}$
# 
# <!-- ### Degree centrality
# $degree\_centrality(v) = d(v)/(n-1),$
#  
# ### Closeness centrality (for connected graph)
# $closeness\_centrality(v) = (n-1)/\sum_{u\neq v}d(u, v)$
# 
# ### Betweenness_centrality (for connected graph)
# $betweenness\_centrality(v) = \frac{2}{(n-1)(n-2)}\sum_{\{s, t\} : s \neq t, s\neq v\neq t} \frac{\vert\{shortest\ path\ btw\ s\ and\ t\ going\ through\ v\}\vert}{\vert\{shortest\ path\ btw\ s\ and\ t\}\vert}$ -->
# 

# ### Computation
# Comments and loads the files if already existing (see next section).

# In[ ]:


print('Compute statistics on graph nodes (topology)...')


# In[ ]:


# Graphs in data set
t1 = time.time()
for G in data_set_G_list:
    networkx.set_node_attributes(G, dict(networkx.degree(G)), 'degree')
    networkx.set_node_attributes(G, networkx.degree_centrality(G), 'degree_centrality')
    networkx.set_node_attributes(G, networkx.closeness_centrality(G), 'closeness_centrality')
    networkx.set_node_attributes(G, networkx.betweenness_centrality(G), 'betweenness_centrality')
t2 = time.time()
print(f'Elapsed time for computing statistics on nodes ({len(data_set_G_list)} graph(s)): {t2-t1:.3g} s')


# In[ ]:


# Generated graphs
t1 = time.time()
for G in G_gen_list:
    networkx.set_node_attributes(G, dict(networkx.degree(G)), 'degree')
    networkx.set_node_attributes(G, networkx.degree_centrality(G), 'degree_centrality')
    networkx.set_node_attributes(G, networkx.closeness_centrality(G), 'closeness_centrality')
    networkx.set_node_attributes(G, networkx.betweenness_centrality(G), 'betweenness_centrality')
t2 = time.time()
print(f'Elapsed time for computing statistics on nodes ({len(G_gen_list)} graph(s)): {t2-t1:.3g} s')


# ### Save to / Load from file (pickle)
# Comments appropriate lines.

# In[ ]:


# Graphs in data set with stats on nodes
filename_data_set_graph_stats_pk = os.path.join(in_dir_gen, f'data_set_graph_list_with_stats_on_nodes.pickle')

# Save pickle
print('Save data set with statistics on nodes (topology)...')
with open(filename_data_set_graph_stats_pk, 'wb') as f: pickle.dump(data_set_G_list, file=f)

# # Load pickle
# print('Load data set with statistics on nodes (topology)...')
# with open(filename_data_set_graph_stats_pk, 'rb') as f: data_set_G_list = pickle.load(f)


# In[ ]:


# Generated graphs with stats on nodes
filename_gen_graph_stats_pk = os.path.join(in_dir_gen, f'gen_graph_list_with_stats_on_nodes.pickle')

# Save pickle
print('Save generated graphs with statistics on nodes (topology)...')
with open(filename_gen_graph_stats_pk, 'wb') as f: pickle.dump(G_gen_list, file=f)

# # Load pickle
# print('Load generated graphs with statistics on nodes (topology)...')
# with open(filename_gen_graph_stats_pk, 'rb') as f: G_gen_list = pickle.load(f)


# ### Plots

# In[ ]:


print('Plot statistics on graph nodes (topology)...')


# In[26]:


figsize = figsize_lh4


# ### Degree

# In[ ]:


# Attribute (key)
attr_stat = 'degree'

# Set distribution
# ----------------
# distribution of each graph
data_distrib = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in data_set_G_list]
gen_distrib  = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in G_gen_list]
# Note: each entry in the lists above is an array of values of length equal to the number of nodes in the corresponding graph

# distribution of all graphs together
data_distrib_all = np.hstack(data_distrib)
gen_distrib_all  = np.hstack(gen_distrib)

# distribution of mean for each graph
data_distrib_mean = np.asarray([d.mean() for d in data_distrib])
gen_distrib_mean  = np.asarray([d.mean() for d in gen_distrib])

# distribution of standard deviation for each graph
data_distrib_std = np.asarray([d.std() for d in data_distrib])
gen_distrib_std  = np.asarray([d.std() for d in gen_distrib])

# some quantiles for every graph separately
quantile = np.array([0, .05, .1, .25, .5, .75, .9, .95, 1.])
label_list = ['min-max', '5-95%', '10-90%', '25-75%']
alpha_list = [.2, .3, .5, .7]
data_distrib_quantile  = np.asarray([np.quantile(d, q=quantile) for d in data_distrib]) # shape (n_graphs, n_quantiles)
gen_distrib_quantile   = np.asarray([np.quantile(d, q=quantile) for d in gen_distrib])  # shape (n_graphs, n_quantiles)

# Plot
# ----
plt.subplots(1,3, figsize=figsize)

plt.subplot(1,3,1)
vmin = min(data_distrib_all.min(), gen_distrib_all.min()) - 0.01
vmax = max(data_distrib_all.max(), gen_distrib_all.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_all, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_all,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : nodes over all graphs', fontsize=10)
plt.legend()
#plt.grid()

plt.subplot(1,3,2)
vmin = min(data_distrib_mean.min(), gen_distrib_mean.min()) - 0.01
vmax = max(data_distrib_mean.max(), gen_distrib_mean.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_mean, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_mean,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : mean of every graph', fontsize=10)
plt.legend()
#plt.grid()

plt.subplot(1,3,3)
vmin = min(data_distrib_std.min(), gen_distrib_std.min()) - 0.01
vmax = max(data_distrib_std.max(), gen_distrib_std.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_std, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_std,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : std of every graph', fontsize=10)
plt.legend()
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_1.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# Plot
plt.subplots(1, 2, figsize=figsize)
vmin = min(data_distrib_quantile.min(), gen_distrib_quantile.min()) - 0.01
vmax = max(data_distrib_quantile.max(), gen_distrib_quantile.max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

plt.subplot(1,2,1)
plt.plot(np.arange(data_distrib_quantile.shape[0]), data_distrib_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(data_distrib_quantile.shape[0]), data_distrib_quantile[:,i], data_distrib_quantile[:,-i-1], color=col_data, alpha=alpha_list[i], label=label_list[i])

plt.legend()
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distrib. for each graph - data set', fontsize=10)

plt.subplot(1,2,2)
plt.plot(np.arange(gen_distrib_quantile.shape[0]), gen_distrib_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(gen_distrib_quantile.shape[0]), gen_distrib_quantile[:,i], gen_distrib_quantile[:,-i-1], color=col_gen, alpha=alpha_list[i], label=label_list[i])

plt.legend()
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distrib. for each graph - generated', fontsize=10)

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_2.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# ### Degree centrality

# In[ ]:


# Attribute (key)
attr_stat = 'degree_centrality'

# Set distribution
# ----------------
# distribution of each graph
data_distrib = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in data_set_G_list]
gen_distrib  = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in G_gen_list]
# Note: each entry in the lists above is an array of values of length equal to the number of nodes in the corresponding graph

# distribution of all graphs together
data_distrib_all = np.hstack(data_distrib)
gen_distrib_all  = np.hstack(gen_distrib)

# distribution of mean for each graph
data_distrib_mean = np.asarray([d.mean() for d in data_distrib])
gen_distrib_mean  = np.asarray([d.mean() for d in gen_distrib])

# distribution of standard deviation for each graph
data_distrib_std = np.asarray([d.std() for d in data_distrib])
gen_distrib_std  = np.asarray([d.std() for d in gen_distrib])

# some quantiles for every graph separately
quantile = np.array([0, .05, .1, .25, .5, .75, .9, .95, 1.])
label_list = ['min-max', '5-95%', '10-90%', '25-75%']
alpha_list = [.2, .3, .5, .7]
data_distrib_quantile  = np.asarray([np.quantile(d, q=quantile) for d in data_distrib]) # shape (n_graphs, n_quantiles)
gen_distrib_quantile   = np.asarray([np.quantile(d, q=quantile) for d in gen_distrib])  # shape (n_graphs, n_quantiles)

# Plot
# ----
plt.subplots(1,3, figsize=figsize)

plt.subplot(1,3,1)
vmin = min(data_distrib_all.min(), gen_distrib_all.min()) - 0.01
vmax = max(data_distrib_all.max(), gen_distrib_all.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_all, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_all,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : nodes over all graphs', fontsize=10)
plt.legend()
#plt.grid()

plt.subplot(1,3,2)
vmin = min(data_distrib_mean.min(), gen_distrib_mean.min()) - 0.01
vmax = max(data_distrib_mean.max(), gen_distrib_mean.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_mean, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_mean,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : mean', fontsize=10)
plt.legend()
#plt.grid()

plt.subplot(1,3,3)
vmin = min(data_distrib_std.min(), gen_distrib_std.min()) - 0.01
vmax = max(data_distrib_std.max(), gen_distrib_std.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_std, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_std,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : std', fontsize=10)
plt.legend()
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_1.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# Plot
plt.subplots(1, 2, figsize=figsize)
vmin = min(data_distrib_quantile.min(), gen_distrib_quantile.min()) - 0.01
vmax = max(data_distrib_quantile.max(), gen_distrib_quantile.max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

plt.subplot(1,2,1)
plt.plot(np.arange(data_distrib_quantile.shape[0]), data_distrib_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(data_distrib_quantile.shape[0]), data_distrib_quantile[:,i], data_distrib_quantile[:,-i-1], color=col_data, alpha=alpha_list[i], label=label_list[i])

plt.legend()
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distrib. for each graph - data set', fontsize=10)

plt.subplot(1,2,2)
plt.plot(np.arange(gen_distrib_quantile.shape[0]), gen_distrib_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(gen_distrib_quantile.shape[0]), gen_distrib_quantile[:,i], gen_distrib_quantile[:,-i-1], color=col_gen, alpha=alpha_list[i], label=label_list[i])

plt.legend()
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distrib. for each graph - generated', fontsize=10)

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_2.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# ### Closeness centrality

# In[ ]:


# Attribute (key)
attr_stat = 'closeness_centrality'

# Set distribution
# ----------------
# distribution of each graph
data_distrib = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in data_set_G_list]
gen_distrib  = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in G_gen_list]
# Note: each entry in the lists above is an array of values of length equal to the number of nodes in the corresponding graph

# distribution of all graphs together
data_distrib_all = np.hstack(data_distrib)
gen_distrib_all  = np.hstack(gen_distrib)

# distribution of mean for each graph
data_distrib_mean = np.asarray([d.mean() for d in data_distrib])
gen_distrib_mean  = np.asarray([d.mean() for d in gen_distrib])

# distribution of standard deviation for each graph
data_distrib_std = np.asarray([d.std() for d in data_distrib])
gen_distrib_std  = np.asarray([d.std() for d in gen_distrib])

# some quantiles for every graph separately
quantile = np.array([0, .05, .1, .25, .5, .75, .9, .95, 1.])
label_list = ['min-max', '5-95%', '10-90%', '25-75%']
alpha_list = [.2, .3, .5, .7]
data_distrib_quantile  = np.asarray([np.quantile(d, q=quantile) for d in data_distrib]) # shape (n_graphs, n_quantiles)
gen_distrib_quantile   = np.asarray([np.quantile(d, q=quantile) for d in gen_distrib])  # shape (n_graphs, n_quantiles)

# Plot
# ----
plt.subplots(1,3, figsize=figsize)

plt.subplot(1,3,1)
vmin = min(data_distrib_all.min(), gen_distrib_all.min()) - 0.01
vmax = max(data_distrib_all.max(), gen_distrib_all.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_all, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_all,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : nodes over all graphs', fontsize=10)
plt.legend()
#plt.grid()

plt.subplot(1,3,2)
vmin = min(data_distrib_mean.min(), gen_distrib_mean.min()) - 0.01
vmax = max(data_distrib_mean.max(), gen_distrib_mean.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_mean, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_mean,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : mean', fontsize=10)
plt.legend()
#plt.grid()

plt.subplot(1,3,3)
vmin = min(data_distrib_std.min(), gen_distrib_std.min()) - 0.01
vmax = max(data_distrib_std.max(), gen_distrib_std.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_std, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_std,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : std', fontsize=10)
plt.legend()
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_1.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# Plot
plt.subplots(1, 2, figsize=figsize)
vmin = min(data_distrib_quantile.min(), gen_distrib_quantile.min()) - 0.01
vmax = max(data_distrib_quantile.max(), gen_distrib_quantile.max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

plt.subplot(1,2,1)
plt.plot(np.arange(data_distrib_quantile.shape[0]), data_distrib_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(data_distrib_quantile.shape[0]), data_distrib_quantile[:,i], data_distrib_quantile[:,-i-1], color=col_data, alpha=alpha_list[i], label=label_list[i])

plt.legend()
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distrib. for each graph - data set', fontsize=10)

plt.subplot(1,2,2)
plt.plot(np.arange(gen_distrib_quantile.shape[0]), gen_distrib_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(gen_distrib_quantile.shape[0]), gen_distrib_quantile[:,i], gen_distrib_quantile[:,-i-1], color=col_gen, alpha=alpha_list[i], label=label_list[i])

plt.legend()
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distrib. for each graph - generated', fontsize=10)

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_2.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# ### Betweenness centrality

# In[ ]:


# Attribute (key)
attr_stat = 'betweenness_centrality'

# Set distribution
# ----------------
# distribution of each graph
data_distrib = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in data_set_G_list]
gen_distrib  = [np.array(list(networkx.get_node_attributes(G, attr_stat).values())) for G in G_gen_list]
# Note: each entry in the lists above is an array of values of length equal to the number of nodes in the corresponding graph

# distribution of all graphs together
data_distrib_all = np.hstack(data_distrib)
gen_distrib_all  = np.hstack(gen_distrib)

# distribution of mean for each graph
data_distrib_mean = np.asarray([d.mean() for d in data_distrib])
gen_distrib_mean  = np.asarray([d.mean() for d in gen_distrib])

# distribution of standard deviation for each graph
data_distrib_std = np.asarray([d.std() for d in data_distrib])
gen_distrib_std  = np.asarray([d.std() for d in gen_distrib])

# some quantiles for every graph separately
quantile = np.array([0, .05, .1, .25, .5, .75, .9, .95, 1.])
label_list = ['min-max', '5-95%', '10-90%', '25-75%']
alpha_list = [.2, .3, .5, .7]
data_distrib_quantile  = np.asarray([np.quantile(d, q=quantile) for d in data_distrib]) # shape (n_graphs, n_quantiles)
gen_distrib_quantile   = np.asarray([np.quantile(d, q=quantile) for d in gen_distrib])  # shape (n_graphs, n_quantiles)

# Plot
# ----
plt.subplots(1,3, figsize=figsize)

plt.subplot(1,3,1)
vmin = min(data_distrib_all.min(), gen_distrib_all.min()) - 0.01
vmax = max(data_distrib_all.max(), gen_distrib_all.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_all, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_all,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : nodes over all graphs', fontsize=10)
plt.legend()
#plt.grid()

plt.subplot(1,3,2)
vmin = min(data_distrib_mean.min(), gen_distrib_mean.min()) - 0.01
vmax = max(data_distrib_mean.max(), gen_distrib_mean.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_mean, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_mean,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : mean', fontsize=10)
plt.legend()
#plt.grid()

plt.subplot(1,3,3)
vmin = min(data_distrib_std.min(), gen_distrib_std.min()) - 0.01
vmax = max(data_distrib_std.max(), gen_distrib_std.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_distrib_std, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_distrib_std,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'{attr_stat} : std', fontsize=10)
plt.legend()
#plt.grid()

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_1.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()

# Plot
plt.subplots(1, 2, figsize=figsize)
vmin = min(data_distrib_quantile.min(), gen_distrib_quantile.min()) - 0.01
vmax = max(data_distrib_quantile.max(), gen_distrib_quantile.max()) + 0.01
ylim = [vmin-0.05*(vmax-vmin), vmax+0.05*(vmax-vmin)]

plt.subplot(1,2,1)
plt.plot(np.arange(data_distrib_quantile.shape[0]), data_distrib_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(data_distrib_quantile.shape[0]), data_distrib_quantile[:,i], data_distrib_quantile[:,-i-1], color=col_data, alpha=alpha_list[i], label=label_list[i])

plt.legend()
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distrib. for each graph - data set', fontsize=10)

plt.subplot(1,2,2)
plt.plot(np.arange(gen_distrib_quantile.shape[0]), gen_distrib_quantile[:,4], c='black', label=f'50%')
for i in range(4):
    plt.fill_between(np.arange(gen_distrib_quantile.shape[0]), gen_distrib_quantile[:,i], gen_distrib_quantile[:,-i-1], color=col_gen, alpha=alpha_list[i], label=label_list[i])

plt.legend()
plt.ylim(ylim)
plt.grid()
plt.title(f'{attr_stat}: distrib. for each graph - generated', fontsize=10)

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_{attr_stat}_2.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# ## Karstnet statistics (global on graph)
# 
# The class `KGraph` has methods to compute statistics (float numbers), and the method `characterize_graph` compute those statistics and store them in a dictionary, as given in the table below.
# 
# The statistics marked with (*) in the description below are computed on the simplified graph, which consists of the graph obtained by removing the nodes of degree 2.
# 
# | method                    | key                        | description |
# |:------                    |:-----                      |:------------|
# |`mean_length`              |'mean length'               | mean length of the branches                                                  |
# |`coef_variation_length`    |'cv length'                 | coefficient of variation (std. dev. / mean) of branch length                 |
# |`length_entropy`           |'length entropy'            | entropy of branch length                                                     |
# |`mean_tortuosity`          |'tortuosity'                | mean tortuosity of branches                                                  |
# |`orientation_entropy`      |'orientation entropy'       | entropy of edges orientation                                                 |
# |`average_SPL`              |'aspl'                      | average shortest path length (*)                                             |
# |`central_point_dominance`  |'cpd'                       | central point dominance (*)                                                  |
# |`mean_degree_and_CV`       |'mean degree'               | mean of node degrees (*)                                                     |
# |                           |'cv degree'                 | coefficient of variation of node degrees (*)                                 |
# |`correlation_vertex_degree`|'correlation vertex degree' | correlation of node degrees over pairs of connected nodes (assortativity) (*)|
# 
# *Notes:* 
# - A branch of a graph is path where all the nodes except the two extremities are nodes of degree 2. The tortuosity of a branch is the ratio of its length over the distance between its two extremities.
# - Average shortest path length is equal to the inverse of the harmonic mean of the closeness centrality of the nodes.
# 

# In[ ]:


print('Compute Karstnet statistics...')


# In[ ]:


# Karstnet statistics for data set
t1 = time.time()
data_mean_length         = np.asarray([KG.mean_length()               for KG in data_set_KG_list])
data_cv_length           = np.asarray([KG.coef_variation_length()     for KG in data_set_KG_list])
data_length_entropy      = np.asarray([KG.length_entropy()            for KG in data_set_KG_list])
data_mean_tortuosity     = np.asarray([KG.mean_tortuosity()           for KG in data_set_KG_list])
data_orientation_entropy = np.asarray([KG.orientation_entropy()       for KG in data_set_KG_list])
data_aspl                = np.asarray([KG.average_SPL()               for KG in data_set_KG_list])
data_cpd                 = np.asarray([KG.central_point_dominance()   for KG in data_set_KG_list])
data_corr_node_degree    = np.asarray([KG.correlation_vertex_degree() for KG in data_set_KG_list])
data_mean_degree, data_CV_degree = np.asarray([KG.mean_degree_and_CV() for KG in data_set_KG_list]).T
t2 = time.time()
print(f'Elapsed time for computing statistics (global) ({len(data_set_KG_list)} graph(s)): {t2-t1:.3g} s')


# In[ ]:


# Karstnet statistics for generated graphs
t1 = time.time()
gen_mean_length         = np.asarray([KG.mean_length()               for KG in KG_gen_list])
gen_cv_length           = np.asarray([KG.coef_variation_length()     for KG in KG_gen_list])
gen_length_entropy      = np.asarray([KG.length_entropy()            for KG in KG_gen_list])
gen_mean_tortuosity     = np.asarray([KG.mean_tortuosity()           for KG in KG_gen_list])
gen_orientation_entropy = np.asarray([KG.orientation_entropy()       for KG in KG_gen_list])
gen_aspl                = np.asarray([KG.average_SPL()               for KG in KG_gen_list])
gen_cpd                 = np.asarray([KG.central_point_dominance()   for KG in KG_gen_list])
gen_corr_node_degree    = np.asarray([KG.correlation_vertex_degree() for KG in KG_gen_list])
gen_mean_degree, gen_CV_degree = np.asarray([KG.mean_degree_and_CV() for KG in KG_gen_list]).T
t2 = time.time()
print(f'Elapsed time for computing statistics (global) ({len(KG_gen_list)} graph(s)): {t2-t1:.3g} s')


# In[33]:


if save_fig_png:
    fig_counter = fig_counter+1


# ### Graphics - histograms

# In[ ]:


print('Plot Karstnet statistics (histogram)...')

# Plot
# ----
figsize = figsize_equal_l6

plt.subplots(4,3, figsize=figsize)

plt.subplot(4,3,1)
vmin = min(data_mean_length.min(), gen_mean_length.min()) - 0.01
vmax = max(data_mean_length.max(), gen_mean_length.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_mean_length, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_mean_length,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'mean length')
plt.legend()
#plt.grid()

plt.subplot(4,3,2)
vmin = min(data_cv_length.min(), gen_cv_length.min()) - 0.01
vmax = max(data_cv_length.max(), gen_cv_length.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_cv_length, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_cv_length,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'cv length')
plt.legend()
#plt.grid()

plt.subplot(4,3,3)
vmin = min(data_length_entropy.min(), gen_length_entropy.min()) - 0.01
vmax = max(data_length_entropy.max(), gen_length_entropy.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_length_entropy, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_length_entropy,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'length entropy')
plt.legend()
#plt.grid()

plt.subplot(4,3,4)
vmin = min(data_mean_tortuosity.min(), gen_mean_tortuosity.min()) - 0.01
vmax = max(data_mean_tortuosity.max(), gen_mean_tortuosity.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_mean_tortuosity, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_mean_tortuosity,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'mean tortuosity')
plt.legend()
#plt.grid()

plt.subplot(4,3,5)
vmin = min(data_orientation_entropy.min(), gen_orientation_entropy.min()) - 0.01
vmax = max(data_orientation_entropy.max(), gen_orientation_entropy.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_orientation_entropy, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_orientation_entropy,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'orientation entropy')
plt.legend()
#plt.grid()

plt.subplot(4,3,6)
vmin = min(data_aspl.min(), gen_aspl.min()) - 0.01
vmax = max(data_aspl.max(), gen_aspl.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_aspl, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_aspl,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'average shortest path length')
plt.legend()
#plt.grid()

plt.subplot(4,3,7)
vmin = min(data_cpd.min(), gen_cpd.min()) - 0.01
vmax = max(data_cpd.max(), gen_cpd.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_cpd, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_cpd,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'central point dominance')
plt.legend()
#plt.grid()

plt.subplot(4,3,8)
vmin = min(data_corr_node_degree.min(), gen_corr_node_degree.min()) - 0.01
vmax = max(data_corr_node_degree.max(), gen_corr_node_degree.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_corr_node_degree, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_corr_node_degree,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'correlation node degree (assortativity)')
plt.legend()
#plt.grid()

plt.subplot(4,3,9)
vmin = min(data_mean_degree.min(), gen_mean_degree.min()) - 0.01
vmax = max(data_mean_degree.max(), gen_mean_degree.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_mean_degree, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_mean_degree,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'mean degree')
plt.legend()
#plt.grid()

plt.subplot(4,3,10)
vmin = min(data_CV_degree.min(), gen_CV_degree.min()) - 0.01
vmax = max(data_CV_degree.max(), gen_CV_degree.max()) + 0.01
nb = 50
bins = np.linspace(vmin, vmax, nb+1)
plt.hist(data_CV_degree, density=True, bins=bins, color=col_data, alpha=.5, label=f'data')
plt.hist(gen_CV_degree,  density=True, bins=bins, color=col_gen,  alpha=.5, label=f'gen.')
plt.title(f'CV degree')
plt.legend()
#plt.grid()

plt.subplot(4,3,11)
plt.axis('off')

plt.subplot(4,3,12)
plt.axis('off')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_KN_summary_hist.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# ### Graphics - boxplots

# In[ ]:


print('Plot Karstnet statistics (boxplot)...')

# Plot
# ----
figsize = figsize_equal_l6

labels = ['data', 'gen']

plt.subplots(4,3, figsize=figsize)

plt.subplot(4,3,1)
data_x = data_mean_length
gen_x  = gen_mean_length
data_x, gen_x = data_x[~np.isnan(data_x)], gen_x[~np.isnan(gen_x)]
bp = plt.boxplot([data_x, gen_x], patch_artist = True, labels=labels, vert=False, medianprops={'color':'black', 'linewidth':1.5})
bp['boxes'][0].set_facecolor(col_data)
bp['boxes'][1].set_facecolor(col_gen)
plt.plot([data_x.mean(), gen_x.mean()], [1, 2], color='black', ls='', marker='x', markersize=10)
plt.grid('x')
plt.title(f'mean length')

plt.subplot(4,3,2)
data_x = data_cv_length
gen_x  = gen_cv_length
data_x, gen_x = data_x[~np.isnan(data_x)], gen_x[~np.isnan(gen_x)]
bp = plt.boxplot([data_x, gen_x], patch_artist = True, labels=labels, vert=False, medianprops={'color':'black', 'linewidth':1.5})
bp['boxes'][0].set_facecolor(col_data)
bp['boxes'][1].set_facecolor(col_gen)
plt.plot([data_x.mean(), gen_x.mean()], [1, 2], color='black', ls='', marker='x', markersize=10)
plt.grid('x')
plt.title(f'cv length')

plt.subplot(4,3,3)
data_x = data_length_entropy
gen_x  = gen_length_entropy
data_x, gen_x = data_x[~np.isnan(data_x)], gen_x[~np.isnan(gen_x)]
bp = plt.boxplot([data_x, gen_x], patch_artist = True, labels=labels, vert=False, medianprops={'color':'black', 'linewidth':1.5})
bp['boxes'][0].set_facecolor(col_data)
bp['boxes'][1].set_facecolor(col_gen)
plt.plot([data_x.mean(), gen_x.mean()], [1, 2], color='black', ls='', marker='x', markersize=10)
plt.grid('x')
plt.title(f'length entropy')

plt.subplot(4,3,4)
data_x = data_mean_tortuosity
gen_x  = gen_mean_tortuosity
data_x, gen_x = data_x[~np.isnan(data_x)], gen_x[~np.isnan(gen_x)]
bp = plt.boxplot([data_x, gen_x], patch_artist = True, labels=labels, vert=False, medianprops={'color':'black', 'linewidth':1.5})
bp['boxes'][0].set_facecolor(col_data)
bp['boxes'][1].set_facecolor(col_gen)
plt.plot([data_x.mean(), gen_x.mean()], [1, 2], color='black', ls='', marker='x', markersize=10)
plt.grid('x')
plt.title(f'mean tortuosity')

plt.subplot(4,3,5)
data_x = data_orientation_entropy
gen_x  = gen_orientation_entropy
data_x, gen_x = data_x[~np.isnan(data_x)], gen_x[~np.isnan(gen_x)]
bp = plt.boxplot([data_x, gen_x], patch_artist = True, labels=labels, vert=False, medianprops={'color':'black', 'linewidth':1.5})
bp['boxes'][0].set_facecolor(col_data)
bp['boxes'][1].set_facecolor(col_gen)
plt.plot([data_x.mean(), gen_x.mean()], [1, 2], color='black', ls='', marker='x', markersize=10)
plt.grid('x')
plt.title(f'orientation entropy')

plt.subplot(4,3,6)
data_x = data_aspl
gen_x  = gen_aspl
data_x, gen_x = data_x[~np.isnan(data_x)], gen_x[~np.isnan(gen_x)]
bp = plt.boxplot([data_x, gen_x], patch_artist = True, labels=labels, vert=False, medianprops={'color':'black', 'linewidth':1.5})
bp['boxes'][0].set_facecolor(col_data)
bp['boxes'][1].set_facecolor(col_gen)
plt.plot([data_x.mean(), gen_x.mean()], [1, 2], color='black', ls='', marker='x', markersize=10)
plt.grid('x')
plt.title(f'average shortest path length')

plt.subplot(4,3,7)
data_x = data_cpd
gen_x  = gen_cpd
data_x, gen_x = data_x[~np.isnan(data_x)], gen_x[~np.isnan(gen_x)]
bp = plt.boxplot([data_x, gen_x], patch_artist = True, labels=labels, vert=False, medianprops={'color':'black', 'linewidth':1.5})
bp['boxes'][0].set_facecolor(col_data)
bp['boxes'][1].set_facecolor(col_gen)
plt.plot([data_x.mean(), gen_x.mean()], [1, 2], color='black', ls='', marker='x', markersize=10)
plt.grid('x')
plt.title(f'central point dominance')

plt.subplot(4,3,8)
data_x = data_corr_node_degree
gen_x  = gen_corr_node_degree
data_x, gen_x = data_x[~np.isnan(data_x)], gen_x[~np.isnan(gen_x)]
bp = plt.boxplot([data_x, gen_x], patch_artist = True, labels=labels, vert=False, medianprops={'color':'black', 'linewidth':1.5})
bp['boxes'][0].set_facecolor(col_data)
bp['boxes'][1].set_facecolor(col_gen)
plt.plot([data_x.mean(), gen_x.mean()], [1, 2], color='black', ls='', marker='x', markersize=10)
plt.grid('x')
plt.title(f'correlation node degree (assortativity)')

plt.subplot(4,3,9)
data_x = data_mean_degree
gen_x  = gen_mean_degree
data_x, gen_x = data_x[~np.isnan(data_x)], gen_x[~np.isnan(gen_x)]
bp = plt.boxplot([data_x, gen_x], patch_artist = True, labels=labels, vert=False, medianprops={'color':'black', 'linewidth':1.5})
bp['boxes'][0].set_facecolor(col_data)
bp['boxes'][1].set_facecolor(col_gen)
plt.plot([data_x.mean(), gen_x.mean()], [1, 2], color='black', ls='', marker='x', markersize=10)
plt.grid('x')
plt.title(f'mean degree')

plt.subplot(4,3,10)
data_x = data_CV_degree
gen_x  = gen_CV_degree
data_x, gen_x = data_x[~np.isnan(data_x)], gen_x[~np.isnan(gen_x)]
bp = plt.boxplot([data_x, gen_x], patch_artist = True, labels=labels, vert=False, medianprops={'color':'black', 'linewidth':1.5})
bp['boxes'][0].set_facecolor(col_data)
bp['boxes'][1].set_facecolor(col_gen)
plt.plot([data_x.mean(), gen_x.mean()], [1, 2], color='black', ls='', marker='x', markersize=10)
plt.grid('x')
plt.title(f'CV degree')

plt.subplot(4,3,11)
plt.axis('off')

plt.subplot(4,3,12)
plt.axis('off')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_KN_summary_boxplot.png')
    fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# ## Multi-Dimensional Scaling (MDS) for karstnet statistics

# In[ ]:


print('Do Multi-Dimensional Scaling (MDS) and plot...')


# In[36]:


# rescaling factor
std_data_mean_length         = np.nanstd(data_mean_length)
std_data_cv_length           = np.nanstd(data_cv_length)
std_data_length_entropy      = np.nanstd(data_length_entropy)
std_data_mean_tortuosity     = np.nanstd(data_mean_tortuosity)
std_data_orientation_entropy = np.nanstd(data_orientation_entropy)
std_data_aspl                = np.nanstd(data_aspl)
std_data_cpd                 = np.nanstd(data_cpd)
std_data_corr_node_degree    = np.nanstd(data_corr_node_degree)
std_data_mean_degree         = np.nanstd(data_mean_degree)
std_data_CV_degree           = np.nanstd(data_CV_degree)


# In[37]:


# rescale measures from data
rescaled_data_mean_length         = data_mean_length         / std_data_mean_length
rescaled_data_cv_length           = data_cv_length           / std_data_cv_length
rescaled_data_length_entropy      = data_length_entropy      / std_data_length_entropy
rescaled_data_mean_tortuosity     = data_mean_tortuosity     / std_data_mean_tortuosity
rescaled_data_orientation_entropy = data_orientation_entropy / std_data_orientation_entropy
rescaled_data_aspl                = data_aspl                / std_data_aspl
rescaled_data_cpd                 = data_cpd                 / std_data_cpd
rescaled_data_corr_node_degree    = data_corr_node_degree    / std_data_corr_node_degree
rescaled_data_mean_degree         = data_mean_degree         / std_data_mean_degree
rescaled_data_CV_degree           = data_CV_degree           / std_data_CV_degree

# rescale measures from gen.
rescaled_gen_mean_length         = gen_mean_length         / std_data_mean_length
rescaled_gen_cv_length           = gen_cv_length           / std_data_cv_length
rescaled_gen_length_entropy      = gen_length_entropy      / std_data_length_entropy
rescaled_gen_mean_tortuosity     = gen_mean_tortuosity     / std_data_mean_tortuosity
rescaled_gen_orientation_entropy = gen_orientation_entropy / std_data_orientation_entropy
rescaled_gen_aspl                = gen_aspl                / std_data_aspl
rescaled_gen_cpd                 = gen_cpd                 / std_data_cpd
rescaled_gen_corr_node_degree    = gen_corr_node_degree    / std_data_corr_node_degree
rescaled_gen_mean_degree         = gen_mean_degree         / std_data_mean_degree
rescaled_gen_CV_degree           = gen_CV_degree           / std_data_CV_degree


# In[38]:


figsize = figsize_equal_def


# #### MDS based on the ten measures, rescaled (wrt. std)

# In[39]:


# Represent each graph as a point in n_stat_measures dimension
# data set graph
data_stats = np.vstack((
    rescaled_data_mean_length,
    rescaled_data_cv_length,
    rescaled_data_length_entropy,
    rescaled_data_mean_tortuosity,
    rescaled_data_orientation_entropy,
    rescaled_data_aspl,
    rescaled_data_cpd,
    rescaled_data_corr_node_degree,
    rescaled_data_mean_degree, 
    rescaled_data_CV_degree)).T

# generated graphs
gen_stats = np.vstack((
    rescaled_gen_mean_length,
    rescaled_gen_cv_length,
    rescaled_gen_length_entropy,
    rescaled_gen_mean_tortuosity,
    rescaled_gen_orientation_entropy,
    rescaled_gen_aspl,
    rescaled_gen_cpd,
    rescaled_gen_corr_node_degree,
    rescaled_gen_mean_degree, 
    rescaled_gen_CV_degree)).T

n_measures = data_stats.shape[1]

# Keep only index where all stat. measures are defined
data_ind = np.where(np.all(~np.isnan(data_stats), axis=1))[0]
gen_ind  = np.where(np.all(~np.isnan(gen_stats), axis=1))[0]

# Do MDS (dim=2) on distance matrix
dmat = scipy.spatial.distance.pdist(np.vstack((data_stats[data_ind, :], gen_stats[gen_ind, :])))
X, f_ssq_mds = multidimensional_scaling(dmat, dim=2)

# Get points in MDS space
data_points = X[:len(data_ind), :]
gen_points  = X[len(data_ind):, :]


# In[ ]:


# Plot points in MDS space
# ------------------------
# col_gen = 'tab:blue'
# col_data = 'tab:orange'

plt.figure(figsize=figsize)
plt.scatter(data_points[:, 0], data_points[:, 1], c=col_data, alpha=.5, s=20, marker='o', label='data')
plt.scatter(gen_points [:, 0], gen_points [:, 1], c=col_gen,  alpha=.5, s=20, marker='o', label='gen')
plt.xlabel(f'MDS - axis 1 ({100*f_ssq_mds[0]:.3g}%)')
plt.ylabel(f'MDS - axis 2 ({100*f_ssq_mds[1]:.3g}%)')
plt.grid()
plt.legend()

plt.title(f'Graphs in MDS space (tot={100*f_ssq_mds.sum():.3g}%) - from all KN measures ({n_measures}) rescaled')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_KN_mds_all_measures_rescaled.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# #### MDS based on the ten measures (not rescaled)

# In[41]:


# Represent each graph as a point in n_stat_measures dimension
# data set graph
data_stats = np.vstack((
    data_mean_length,
    data_cv_length,
    data_length_entropy,
    data_mean_tortuosity,
    data_orientation_entropy,
    data_aspl,
    data_cpd,
    data_corr_node_degree,
    data_mean_degree, 
    data_CV_degree)).T

# generated graphs
gen_stats = np.vstack((
    gen_mean_length,
    gen_cv_length,
    gen_length_entropy,
    gen_mean_tortuosity,
    gen_orientation_entropy,
    gen_aspl,
    gen_cpd,
    gen_corr_node_degree,
    gen_mean_degree, 
    gen_CV_degree)).T

n_measures = data_stats.shape[1]

# Keep only index where all stat. measures are defined
data_ind = np.where(np.all(~np.isnan(data_stats), axis=1))[0]
gen_ind  = np.where(np.all(~np.isnan(gen_stats), axis=1))[0]

# Do MDS (dim=2) on distance matrix
dmat = scipy.spatial.distance.pdist(np.vstack((data_stats[data_ind, :], gen_stats[gen_ind, :])))
X, f_ssq_mds = multidimensional_scaling(dmat, dim=2)

# Get points in MDS space
data_points = X[:len(data_ind), :]
gen_points  = X[len(data_ind):, :]


# In[ ]:


# Plot points in MDS space
# ------------------------
# col_gen = 'tab:blue'
# col_data = 'tab:orange'

plt.figure(figsize=figsize)
plt.scatter(data_points[:, 0], data_points[:, 1], c=col_data, alpha=.5, s=20, marker='o', label='data')
plt.scatter(gen_points [:, 0], gen_points [:, 1], c=col_gen,  alpha=.5, s=20, marker='o', label='gen')
plt.xlabel(f'MDS - axis 1 ({100*f_ssq_mds[0]:.3g}%)')
plt.ylabel(f'MDS - axis 2 ({100*f_ssq_mds[1]:.3g}%)')
plt.grid()
plt.legend()

plt.title(f'Graphs in MDS space (tot={100*f_ssq_mds.sum():.3g}%) - from all KN measures ({n_measures}) not rescaled')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_KN_mds_all_measures_notrescaled.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# #### MDS based on selected measures

# In[43]:


# Represent each graph as a point in n_stat_measures dimension
# data set graph
data_stats = np.vstack((
    # data_mean_length,
    data_cv_length,
    data_length_entropy,
    # data_mean_tortuosity,
    data_orientation_entropy,
    data_aspl,
    data_cpd,
    data_corr_node_degree,
    data_mean_degree, 
    data_CV_degree)).T

# generated graphs
gen_stats = np.vstack((
    # gen_mean_length,
    gen_cv_length,
    gen_length_entropy,
    # gen_mean_tortuosity,
    gen_orientation_entropy,
    gen_aspl,
    gen_cpd,
    gen_corr_node_degree,
    gen_mean_degree, 
    gen_CV_degree)).T

n_measures = data_stats.shape[1]

# Keep only index where all stat. measures are defined
data_ind = np.where(np.all(~np.isnan(data_stats), axis=1))[0]
gen_ind  = np.where(np.all(~np.isnan(gen_stats), axis=1))[0]

# Do MDS (dim=2) on distance matrix
dmat = scipy.spatial.distance.pdist(np.vstack((data_stats[data_ind, :], gen_stats[gen_ind, :])))
X, f_ssq_mds = multidimensional_scaling(dmat, dim=2)

# Get points in MDS space
data_points = X[:len(data_ind), :]
gen_points  = X[len(data_ind):, :]


# In[ ]:


# Plot points in MDS space
# ------------------------
# col_gen = 'tab:blue'
# col_data = 'tab:orange'

plt.figure(figsize=figsize)
plt.scatter(data_points[:, 0], data_points[:, 1], c=col_data, alpha=.5, s=20, marker='o', label='data')
plt.scatter(gen_points [:, 0], gen_points [:, 1], c=col_gen,  alpha=.5, s=20, marker='o', label='gen')
plt.xlabel(f'MDS - axis 1 ({100*f_ssq_mds[0]:.3g}%)')
plt.ylabel(f'MDS - axis 2 ({100*f_ssq_mds[1]:.3g}%)')
plt.grid()
plt.legend()

plt.title(f'Graphs in MDS space (tot={100*f_ssq_mds.sum():.3g}%) - from selected KN measures ({n_measures}) not rescaled')

if save_fig_png:
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_stats_KN_mds_selected_measures_notrescaled.png')
    # fig_counter = fig_counter+1

if plt_show:
    plt.show()
else:
    plt.close()


# In[45]:


if save_fig_png:
    fig_counter = fig_counter+1

