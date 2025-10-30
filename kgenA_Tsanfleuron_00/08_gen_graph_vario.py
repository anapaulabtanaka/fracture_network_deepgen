#!/usr/bin/env python
# coding: utf-8

# # Statistics - varriogram along graph branches
# 
# *Note: only for graph with at least one node feature (in addition to position).*

# In[ ]:


import networkx
# import torch, torch_geometric
import numpy as np
# import scipy
import matplotlib.pyplot as plt
# import time

# import pyvista as pv
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
# # from graph_rnn import *
# # from graph_ddpm import *
# # from ml_utils import *
# from general_utils import *
# from graph_plot import *
# # from magic_utils import *
 
with open('../utils/graph_utils.py') as f: exec(f.read())
# with open('../utils/graph_rnn.py') as f: exec(f.read())
# with open('../utils/graph_ddpm.py') as f: exec(f.read())
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
# off_screen = True    # PARAMS (show graphics 3D ?)

save_fig_png = True  # PARAMS
fig_prefix = '08'    # PARAMS

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


# ### Load list of generated graphs

# In[ ]:


print('Load generated graphs...')

in_dir_gen = 'out_gen_graph' 
filename_gen_graph_pk = os.path.join(in_dir_gen, f'gen_graph_list.pickle')

# Load list of generated graph (G_gen_list: of generated graphs in networkx format)
with open(filename_gen_graph_pk, 'rb') as f: G_gen_list = pickle.load(f)


# ### Set in "Karsnet format"

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


# ## Statistics - variogram along branches

# In[ ]:


print('Compute and plot variogram along branches...')


# In[10]:


# Colors for further graphs
col_gen = 'tab:blue'
col_data = 'tab:orange'

col2_gen = 'darkblue'
col2_data = 'tab:red'


# In[11]:


# Definition of quantiles used further
quant = (0., 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.)
quant_name = [f'{100*q:.3g}%' for q in quant]
nquant = len(quant)
mquant = nquant//2
alpha_list = np.linspace(.2,.7, mquant)

# figsize...
figsize = figsize_lh3


# ### All experimental variograms

# In[ ]:


# # Parameters for the classes of experimental variogram
# ncla = 18                 # number of classes
# cla_base_length = 30      # base length for each class (space btw class center)
# cla_length_factor = 1.5   # factor by which multiply base length to get the class length
# cla_center = cla_base_length*(0.5+np.arange(ncla))
# cla_length = cla_length_factor*cla_base_length

# Parameters for the classes of experimental variogram
cla_limit = np.array([0., 2., 4., 8., 12., 20., 50., 80., 110., 150., 200., 250., 300.])
cla_center = 0.5*(cla_limit[:-1]+cla_limit[1:])
cla_length = np.diff(cla_limit)
ncla = len(cla_center)

# Experimental variograms 
# # - if variogram cloud already computed
# data_vario_exp = [KG.variogram_exp_along_branches(cla_center=cla_center, cla_length=cla_length, variogramCloud=vario_cloud, make_plot=False) for KG, vario_cloud in zip(data_set_KG_list, data_vario_cloud)]
# gen_vario_exp  = [KG.variogram_exp_along_branches(cla_center=cla_center, cla_length=cla_length, variogramCloud=vario_cloud, make_plot=False) for KG, vario_cloud in zip(KG_gen_list,      gen_vario_cloud)]
# - otherwise
data_vario_exp = [KG.variogram_exp_along_branches(cla_center=cla_center, cla_length=cla_length, make_plot=False) for KG in data_set_KG_list]
gen_vario_exp  = [KG.variogram_exp_along_branches(cla_center=cla_center, cla_length=cla_length, make_plot=False) for KG in KG_gen_list]


# In[13]:


# Extract h (absissa), g (ordinate) and c (counter) for every experimental variogram

# data set
data_hexp = np.full((len(data_vario_exp), ncla), np.nan)
data_gexp = np.full((attr_ncomp, len(data_vario_exp), ncla), np.nan) # axis 0: attribute index
data_cexp = np.full((len(data_vario_exp), ncla), np.nan)
for i, (hexp, gexp, cexp) in enumerate(data_vario_exp):
    data_hexp[i] = hexp
    data_gexp[:, i, :] = gexp.T
    data_cexp[i] = cexp

# generated graph
gen_hexp = np.full((len(gen_vario_exp), ncla), np.nan)
gen_gexp = np.full((attr_ncomp, len(gen_vario_exp), ncla), np.nan) # axis 0: attribute index
gen_cexp = np.full((len(gen_vario_exp), ncla), np.nan)
for i, (hexp, gexp, cexp) in enumerate(gen_vario_exp):
    gen_hexp[i] = hexp
    gen_gexp[:, i, :] = gexp.T
    gen_cexp[i] = cexp


# In[ ]:


out_name = 'KN_branch_vario_exp_all'

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot
    # ----
    plt.figure(figsize=figsize)
    for hexp, gexp, cexp in zip(data_hexp, data_gexp[i_attr], data_cexp):
        plt.plot(hexp, gexp, c=col_data, alpha=.1)
    for hexp, gexp, cexp in zip(gen_hexp, gen_gexp[i_attr], gen_cexp):
        plt.plot(hexp, gexp, c=col_gen, alpha=.1)
    plt.plot([hexp[0], hexp[0]+.1], [np.nan, np.nan], c=col_data, label=f'data') # for legend
    plt.plot([hexp[0], hexp[0]+.1], [np.nan, np.nan], c=col_gen,  label=f'gen.')  # for legend
    plt.legend()
    plt.grid()
    plt.xlabel('h')
    plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')

    plt.title(f'Experimental variograms along branches for {attr_label}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1


# ### Average experimental variogram

# In[15]:


data_hexp_mean = np.nanmean(data_hexp, axis=0)
data_gexp_mean = np.nanmean(data_gexp, axis=1)
data_cexp_mean = np.mean(data_cexp, axis=0)

gen_hexp_mean = np.nanmean(gen_hexp, axis=0)
gen_gexp_mean = np.nanmean(gen_gexp, axis=1)
gen_cexp_mean = np.mean(gen_cexp, axis=0)


# In[ ]:


out_name = 'KN_branch_vario_exp_average'

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot
    # ----
    plt.figure(figsize=figsize)
    
    plt.plot(data_hexp_mean, data_gexp_mean[i_attr], marker='+', c=col_data, label=f'data')
    for i, c in enumerate(data_cexp_mean):
        if c > 0:
            plt.text(data_hexp_mean[i], data_gexp_mean[i_attr, i], f'{c:4.1f}', c=col_data, ha='left', va='top')

    plt.plot(gen_hexp_mean,  gen_gexp_mean[i_attr], marker='+',  c=col_gen, label=f'gen.')
    for i, c in enumerate(gen_cexp_mean):
        if c > 0:
            plt.text(gen_hexp_mean[i], gen_gexp_mean[i_attr, i], f'{c:4.1f}', c=col_gen, ha='left', va='top')

    plt.legend()
    plt.grid()
    plt.xlabel('h')
    plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')

    plt.title(f'Average experimental variogram along branches for {attr_label}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1


# ### Experimental variogram : mean +/ std.

# In[17]:


## +/- standard dev.
data_hexp_std = np.nanstd(data_hexp, axis=0)
data_gexp_std = np.nanstd(data_gexp, axis=1)
data_cexp_std = np.std(data_cexp, axis=0)

gen_hexp_std = np.nanstd(gen_hexp, axis=0)
gen_gexp_std = np.nanstd(gen_gexp, axis=1)
gen_cexp_std = np.std(gen_cexp, axis=0)


# In[ ]:


out_name = 'KN_branch_vario_exp_mean_std'

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot
    # ----
    plt.figure(figsize=figsize)
    
    plt.plot(data_hexp_mean, data_gexp_mean[i_attr], ls='dashed', c=col_data, label=f'data')
    for hm, hs, gm, gs in zip(data_hexp_mean, data_hexp_std, data_gexp_mean[i_attr], data_gexp_std[i_attr]):
        if np.all(~np.isnan([hm, hs, gm])):
            plt.plot([hm-hs, hm+hs], [gm, gm], ls='solid', lw=1, marker='+', markersize=10, c=col_data)
        if np.all(~np.isnan([hm, gm, gs])):
            plt.plot([hm, hm], [gm-gs, gm+gs], ls='solid', lw=1, marker='+', markersize=10, c=col_data)
    for i, (cm, cs) in enumerate(zip(data_cexp_mean, data_cexp_std)):
        if cm > 0:
            plt.text(data_hexp_mean[i], data_gexp_mean[i_attr, i], f'{cm:4.1f} +/- {cs:4.1f}', c=col_data, ha='left', va='top')

    plt.plot(gen_hexp_mean, gen_gexp_mean[i_attr], ls='dashed', c=col_gen, label=f'gen.')
    for hm, hs, gm, gs in zip(gen_hexp_mean, gen_hexp_std, gen_gexp_mean[i_attr], gen_gexp_std[i_attr]):
        if np.all(~np.isnan([hm, hs, gm])):
            plt.plot([hm-hs, hm+hs], [gm, gm], ls='solid', lw=1, marker='+', markersize=10, c=col_gen)
        if np.all(~np.isnan([hm, gm, gs])):
            plt.plot([hm, hm], [gm-gs, gm+gs], ls='solid', lw=1, marker='+', markersize=10, c=col_gen)
    for i, (cm, cs) in enumerate(zip(gen_cexp_mean, gen_cexp_std)):
        if cm > 0:
            plt.text(gen_hexp_mean[i], gen_gexp_mean[i_attr, i], f'{cm:4.1f} +/- {cs:4.1f}', c=col_gen, ha='left', va='top')

    plt.legend()
    plt.grid()
    plt.xlabel('h')
    plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')

    plt.title(f'Exp. variogram along branches (mean +/- std.) for {attr_label}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1


# ### Quantile on experimental variograms

# In[19]:


# Compute quantiles 
data_gexp_quant = np.nanquantile(data_gexp, q=quant, axis=1).transpose(1,0,2)
gen_gexp_quant  = np.nanquantile(gen_gexp,  q=quant, axis=1).transpose(1,0,2)


# In[ ]:


out_name = 'KN_branch_vario_exp_quantile_1'

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot quantiles
    # --------------
    plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Variogram - Quantile for data set
    plt.subplot(1,2,1)
    for i in range(mquant):
        plt.fill_between(cla_center, data_gexp_quant[i_attr, i], data_gexp_quant[i_attr, nquant-1-i], color=col_data,
                         alpha=alpha_list[i], label=f'{quant_name[i]}-{quant_name[nquant-1-i]}')
    plt.plot(cla_center, data_gexp_quant[i_attr, mquant], linestyle='solid',  color=col2_data, label=f'{quant_name[mquant]}')
    plt.plot(cla_center, data_gexp_mean[i_attr],          linestyle='dashed', color=col2_data, label=f'mean')
    plt.grid()
    plt.legend()
    plt.title(f'data set')

    # Variogram - Quantile for generated graphs
    plt.subplot(1,2,2)
    for i in range(mquant):
        plt.fill_between(cla_center, gen_gexp_quant[i_attr, i], gen_gexp_quant[i_attr, nquant-1-i], color=col_gen,
                         alpha=alpha_list[i], label=f'{quant_name[i]}-{quant_name[nquant-1-i]}')
    plt.plot(cla_center, gen_gexp_quant[i_attr, mquant], linestyle='solid',  color=col2_gen, label=f'{quant_name[mquant]}')
    plt.plot(cla_center, gen_gexp_mean[i_attr],          linestyle='dashed', color=col2_gen, label=f'mean')
    plt.grid()
    plt.legend()
    plt.title(f'Generated graphs')

    plt.suptitle(f'Quantile - exp. vario along branches for {attr_label}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

# if save_fig_png:
#     fig_counter = fig_counter+1



# In[ ]:


out_name = 'KN_branch_vario_exp_quantile_2'

figsize_m = (figsize[0], .6*mquant*figsize[1])

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot quantiles
    # --------------
    plt.subplots(mquant, 1, figsize=figsize_m, sharex=True)

    for i in range(mquant):
        plt.subplot(mquant,1,i+1)

        # Variogram - Quantile for data set
        plt.fill_between(cla_center, data_gexp_quant[i_attr, i], data_gexp_quant[i_attr, nquant-1-i], color=col_data,
                         alpha=alpha_list[i], label=f'data')
        plt.plot(cla_center, data_gexp_quant[i_attr, mquant], linestyle='solid',  color=col2_data, label=f'data - {quant_name[mquant]}')
        plt.plot(cla_center, data_gexp_mean[i_attr],          linestyle='dashed', color=col2_data, label=f'data - mean')

        # Variogram - Quantile for generated graphs
        plt.fill_between(cla_center, gen_gexp_quant[i_attr, i], gen_gexp_quant[i_attr, nquant-1-i], color=col_gen,
                         alpha=alpha_list[i], label=f'gen.')
        plt.plot(cla_center, gen_gexp_quant[i_attr, mquant], linestyle='solid',  color=col2_gen, label=f'gen. - {quant_name[mquant]}')
        plt.plot(cla_center, gen_gexp_mean[i_attr],          linestyle='dashed', color=col2_gen, label=f'gen. - mean')    
        
        plt.grid()
        plt.legend()
        plt.title(f'Exp. vario. along branches for {attr_label} - quantile {quant_name[i]}-{quant_name[nquant-1-i]}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1


# ### Global experimental variogram (from all variogram clouds together)

# In[22]:


# Variogram cloud
data_vario_cloud = [KG.variogram_cloud_along_branches(make_plot=False) for KG in data_set_KG_list]
gen_vario_cloud  = [KG.variogram_cloud_along_branches(make_plot=False) for KG in KG_gen_list]

# Gathering variogram clouds
data_vario_cloud_all_h = np.empty(shape=(0), dtype=float)
data_vario_cloud_all_g = np.empty(shape=(0, attr_ncomp), dtype=float)

for h, g in data_vario_cloud:
    data_vario_cloud_all_h = np.hstack((data_vario_cloud_all_h, h))
    data_vario_cloud_all_g = np.vstack((data_vario_cloud_all_g, g))

gen_vario_cloud_all_h = np.empty(shape=(0), dtype=float)
gen_vario_cloud_all_g = np.empty(shape=(0, attr_ncomp), dtype=float)

for h, g in gen_vario_cloud:
    gen_vario_cloud_all_h = np.hstack((gen_vario_cloud_all_h, h))
    gen_vario_cloud_all_g = np.vstack((gen_vario_cloud_all_g, g))

# Global experimental variogram (from  all variogram clouds together)
KG_fake = kn.KGraph([],{}, verbose=False)
data_hexp_all, data_gexp_all, data_cexp_all = KG_fake.variogram_exp_along_branches(cla_center=cla_center, cla_length=cla_length, variogramCloud=(data_vario_cloud_all_h, data_vario_cloud_all_g), make_plot=False)
gen_hexp_all,  gen_gexp_all,  gen_cexp_all  = KG_fake.variogram_exp_along_branches(cla_center=cla_center, cla_length=cla_length, variogramCloud=(gen_vario_cloud_all_h,  gen_vario_cloud_all_g),  make_plot=False)

data_gexp_all = data_gexp_all.T
gen_gexp_all = gen_gexp_all.T


# In[ ]:


out_name = 'KN_branch_vario_exp_global'

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot
    # ----
    plt.figure(figsize=figsize)

    plt.plot(cla_center, data_gexp_all[i_attr], marker='+', c=col_data, label=f'data - all')
    for i, c in enumerate(data_cexp_all):
        if c > 0:
            plt.text(cla_center[i], data_gexp_all[i_attr, i], f'{c:6.0f}', c=col_data, ha='left', va='top')

    plt.plot(cla_center, gen_gexp_all[i_attr], marker='+', c=col_gen, label=f'gen. - all')
    for i, c in enumerate(gen_cexp_all):
        if c > 0:
            plt.text(cla_center[i], gen_gexp_all[i_attr, i], f'{c:6.0f}', c=col_gen, ha='left', va='top')

    plt.legend()
    plt.grid()
    plt.xlabel('h')
    plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')

    plt.title(f'"Global" experimental variogram along branches for {attr_label}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1


# ### All kernel estimate variograms
# 
# Kernel estimate from the variogram clouds.

# In[24]:


# Variogram cloud
data_vario_cloud = [KG.variogram_cloud_along_branches(make_plot=False) for KG in data_set_KG_list]
gen_vario_cloud  = [KG.variogram_cloud_along_branches(make_plot=False) for KG in KG_gen_list]


# In[25]:


# h0 = max(
#     max([vario_cloud[0].min() for vario_cloud in data_vario_cloud]),
#     max([vario_cloud[0].min() for vario_cloud in gen_vario_cloud])
# )
# h1 = min(
#     min([vario_cloud[0].max() for vario_cloud in data_vario_cloud]),
#     min([vario_cloud[0].max() for vario_cloud in gen_vario_cloud])
# )

h0 = min(
    min([vario_cloud[0].min() for vario_cloud in data_vario_cloud]),
    min([vario_cloud[0].min() for vario_cloud in gen_vario_cloud])
)
h1 = max(
    max([vario_cloud[0].max() for vario_cloud in data_vario_cloud]),
    max([vario_cloud[0].max() for vario_cloud in gen_vario_cloud])
)

hh = np.linspace(h0, h1, 300)

gg_data_vario_ke = [np.asarray([nw_kernel_estimate(hh, h, g[:,i_attr]) for h, g in data_vario_cloud]) for i_attr in range(attr_ncomp)]
gg_gen_vario_ke  = [np.asarray([nw_kernel_estimate(hh, h, g[:,i_attr]) for h, g in gen_vario_cloud] ) for i_attr in range(attr_ncomp)]


# In[ ]:


out_name = 'KN_branch_vario_ke_all'

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot
    # ----
    plt.figure(figsize=figsize)
    for gg in gg_data_vario_ke[i_attr]:
        plt.plot(hh, gg, c=col_data, alpha=.1)
    for gg in gg_gen_vario_ke[i_attr]:
        plt.plot(hh, gg, c=col_gen,  alpha=.1)
    plt.plot([hh[0], hh[0]+.1], [np.nan, np.nan], c=col_data, label=f'data') # for legend
    plt.plot([hh[0], hh[0]+.1], [np.nan, np.nan], c=col_gen,  label=f'gen.') # for legend
    plt.legend()
    plt.grid()
    plt.xlabel('h')
    plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')

    plt.title(f'Kernel estimate variograms along branches for {attr_label}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1


# ### Average kernel estimate variogram

# In[ ]:


gg_data_vario_ke_mean = [np.nanmean(gg_data_vario_ke[i_attr], axis=0) for i_attr in range(attr_ncomp)]
gg_gen_vario_ke_mean  = [np.nanmean(gg_gen_vario_ke[i_attr],  axis=0) for i_attr in range(attr_ncomp)]


# In[ ]:


out_name = 'KN_branch_vario_ke_average'

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot
    # ----
    plt.figure(figsize=figsize)
    
    plt.plot(hh, gg_data_vario_ke_mean[i_attr], c=col_data, label=f'data')
    plt.plot(hh, gg_gen_vario_ke_mean[i_attr],  c=col_gen, label=f'gen.')

    plt.legend()
    plt.grid()
    plt.xlabel('h')
    plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')

    plt.title(f'Average kernel estimate variogram along branches for {attr_label}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1


# ### Quantile on kernel estimate variograms

# In[ ]:


# Compute quantiles 
gg_data_vario_ke_quant = [np.nanquantile(gg_data_vario_ke[i_attr], q=quant, axis=0) for i_attr in range(attr_ncomp)]
gg_gen_vario_ke_quant  = [np.nanquantile(gg_gen_vario_ke[i_attr],  q=quant, axis=0) for i_attr in range(attr_ncomp)]


# In[ ]:


out_name = 'KN_branch_vario_ke_quantile_1'

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot quantiles
    # --------------
    plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Variogram - Quantile for data set
    plt.subplot(1,2,1)
    for i in range(mquant):
        plt.fill_between(hh, gg_data_vario_ke_quant[i_attr][i], gg_data_vario_ke_quant[i_attr][nquant-1-i], color=col_data, 
                         alpha=alpha_list[i], label=f'{quant_name[i]}-{quant_name[nquant-1-i]}')
    plt.plot(hh, gg_data_vario_ke_quant[i_attr][mquant], linestyle='solid',  color=col2_data, label=f'{quant_name[mquant]}')
    plt.plot(hh, gg_data_vario_ke_mean[i_attr],          linestyle='dashed', color=col2_data, label=f'mean')    
    plt.grid()
    plt.legend()
    plt.title(f'data set')

    # Variogram - Quantile for generated graphs
    plt.subplot(1,2,2)
    for i in range(mquant):
        plt.fill_between(hh, gg_gen_vario_ke_quant[i_attr][i], gg_gen_vario_ke_quant[i_attr][nquant-1-i], color=col_gen, 
                        alpha=alpha_list[i], label=f'{quant_name[i]}-{quant_name[nquant-1-i]}')
    plt.plot(hh, gg_gen_vario_ke_quant[i_attr][mquant], linestyle='solid',  color=col2_gen, label=f'{quant_name[mquant]}')
    plt.plot(hh, gg_gen_vario_ke_mean[i_attr],          linestyle='dashed', color=col2_gen, label=f'mean')    
    plt.grid()
    plt.legend()
    plt.title(f'Generated graphs')

    plt.suptitle(f'Quantile - kernel estimate vario along branches for {attr_label}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

# if save_fig_png:
#     fig_counter = fig_counter+1


# In[ ]:


out_name = 'KN_branch_vario_ke_quantile_2'

figsize_m = (figsize[0], .6*mquant*figsize[1])

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot quantiles
    # --------------
    plt.subplots(mquant, 1, figsize=figsize_m, sharex=True)

    for i in range(mquant):
        plt.subplot(mquant,1,i+1)

        # Variogram - Quantile for data set
        plt.fill_between(hh, gg_data_vario_ke_quant[i_attr][i], gg_data_vario_ke_quant[i_attr][nquant-1-i], color=col_data, 
                         alpha=alpha_list[i], label=f'data')
        plt.plot(hh, gg_data_vario_ke_quant[i_attr][mquant], linestyle='solid',  color=col2_data, label=f'data - {quant_name[mquant]}')
        plt.plot(hh, gg_data_vario_ke_mean[i_attr],          linestyle='dashed', color=col2_data, label=f'data - mean')    

        # Variogram - Quantile for generated graphs
        plt.fill_between(hh, gg_gen_vario_ke_quant[i_attr][i], gg_gen_vario_ke_quant[i_attr][nquant-1-i], color=col_gen, 
                         alpha=alpha_list[i], label=f'gen.')
        plt.plot(hh, gg_gen_vario_ke_quant[i_attr][mquant], linestyle='solid',  color=col2_gen, label=f'gen. - {quant_name[mquant]}')    
        plt.plot(hh, gg_gen_vario_ke_mean[i_attr],          linestyle='dashed', color=col2_gen, label=f'gen. - mean')    
        
        plt.grid()
        plt.legend()
        plt.title(f'Kernel estimate vario. along branches for {attr_label} - quantile {quant_name[i]}-{quant_name[nquant-1-i]}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1


# ### Global kernel estimate variogram (from all variogram clouds together)

# In[32]:


# # Variogram cloud
# data_vario_cloud = [KG.variogram_cloud_along_branches(make_plot=False) for KG in data_set_KG_list]
# gen_vario_cloud  = [KG.variogram_cloud_along_branches(make_plot=False) for KG in KG_gen_list]

# Gathering variogram clouds
data_vario_cloud_all_h = np.empty(shape=(0), dtype=float)
data_vario_cloud_all_g = np.empty(shape=(0, attr_ncomp), dtype=float)

for h, g in data_vario_cloud:
    data_vario_cloud_all_h = np.hstack((data_vario_cloud_all_h, h))
    data_vario_cloud_all_g = np.vstack((data_vario_cloud_all_g, g))

gen_vario_cloud_all_h = np.empty(shape=(0), dtype=float)
gen_vario_cloud_all_g = np.empty(shape=(0, attr_ncomp), dtype=float)

for h, g in gen_vario_cloud:
    gen_vario_cloud_all_h = np.hstack((gen_vario_cloud_all_h, h))
    gen_vario_cloud_all_g = np.vstack((gen_vario_cloud_all_g, g))


# In[33]:


h0 = min(data_vario_cloud_all_h.min(), gen_vario_cloud_all_h.min())
h1 = max(data_vario_cloud_all_h.max(), gen_vario_cloud_all_h.max())

hh = np.linspace(h0, h1, 300)

gg_data_vario_ke_all = [nw_kernel_estimate(hh, data_vario_cloud_all_h, data_vario_cloud_all_g[:, i_attr]) for i_attr in range(attr_ncomp)]
gg_gen_vario_ke_all  = [nw_kernel_estimate(hh, gen_vario_cloud_all_h,  gen_vario_cloud_all_g[:, i_attr] ) for i_attr in range(attr_ncomp)]


# In[ ]:


out_name = 'KN_branch_vario_ke_global'

for i_attr in range(attr_ncomp):
    # loop on attributes
    attr_label = attr_label_list[i_attr]
    
    # Plot
    # ----
    plt.figure(figsize=figsize)

    plt.plot(hh, gg_data_vario_ke_all[i_attr], c=col_data, alpha=1, label=f'data')
    plt.plot(hh, gg_gen_vario_ke_all[i_attr],  c=col_gen,  alpha=1, label=f'gen.')

    plt.legend()
    plt.grid()
    plt.xlabel('h')
    plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')

    plt.title(f'"Global" kernel estimate vario. along branches for {attr_label}')

    if save_fig_png:
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/{fig_prefix}_{fig_counter:02d}_{out_name}_{attr}_{i_attr}.png')
        # fig_counter = fig_counter+1

    if plt_show:
        plt.show()
    else:
        plt.close()

if save_fig_png:
    fig_counter = fig_counter+1

