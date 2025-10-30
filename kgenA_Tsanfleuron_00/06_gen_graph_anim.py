#!/usr/bin/env python
# coding: utf-8

# # Graph generation with node features - animation

# In[1]:


import networkx
import torch, torch_geometric
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

import pyvista as pv
import os

import imageio

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

# In[ ]:


print('Define output settings...')

# Output directory (for saving)
# -----------------------------
fig_dir = 'fig'      # PARAMS

plt_show = False     # PARAMS (show graphics 2D ?)
off_screen = True    # PARAMS (show graphics 3D ?)

save_fig_png = True  # PARAMS
fig_prefix = '06'    # PARAMS

fig_counter = 0

if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)


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


# ## Generation of one graph - with animation

# In[ ]:


print('Generation of one graph with animation...')

# Number of graphs
n_graph = 1

# Generate topology
max_n_nodes = 10000 # should not be reached...
# max_n_nodes = 200
min_n_nodes = 5 # will re-draw graph(s) if fewer nodes

torch.random.manual_seed(563)

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
print(f'Number of nodes in generated graph: {G_gen_list[0].number_of_nodes()}')
print(f'Elapsed time for generating {n_graph} graph(s) - topology: {t2-t1:.3g} s')


# In[ ]:


# Generate node features (inplace)
torch.random.manual_seed(357)

end_rescale = node_features_scale_factor_inv
end_center = node_features_shift_inv

t1 = time.time()
G, x_hat_all = generate_graph_node_features(
            G_gen_list[0], 
            ddpm, 
            attr='x',
            end_rescale=end_rescale,
            end_center=end_center, 
            return_intermediate=True,
            device=torch.device('cuda:0'))
x_hat_all = [x.numpy() for x in x_hat_all]
t2 = time.time()

print(f'Elapsed time for generating {n_graph} graph(s) - node features: {t2-t1:.3g} s')


# In[15]:


# Set networkx representation of generated graph with position and other attributes separated
v = np.asarray(list(networkx.get_node_attributes(G, 'x').values()))
remove_node_attribute(G, 'x')

dict_pos = {i:vi[:dim].tolist() for i, vi in enumerate(v)}
networkx.set_node_attributes(G, dict_pos, 'pos')

if attr is not None:
    dict_attr = {i:vi[dim:].tolist() for i, vi in enumerate(v)}
    networkx.set_node_attributes(G, dict_attr, attr)


# ### Animation 2D

# In[ ]:


print('Plot result (2D)...')

# Plot result - 2D
# ================
kwds = kwds_single.copy()

figsize = figsize_single
# -----

out_name = 'anim_final'
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


print('Plot result with details (2D)...')

# Plot result with details
# ========================

# ----- Settings --------------
t_show = np.hstack(
    [np.linspace(                     0, int(0.50*ddpm.n_steps), 6).astype(int),
     np.linspace(int(0.50*ddpm.n_steps), int(0.75*ddpm.n_steps), 7).astype(int)[1:],
     np.linspace(int(0.75*ddpm.n_steps), int(0.90*ddpm.n_steps), 7).astype(int)[1:],
     np.linspace(int(0.90*ddpm.n_steps), ddpm.n_steps          , 7).astype(int)[1:]
    ]).tolist()
# -----------------------------

# Set list of graphs to show
G_list_show = [G.copy() for _ in range(len(t_show))]
for k, t in enumerate(t_show):
    x = x_hat_all[t]

    dict_pos = {i: xi[:dim].tolist() for i, xi in enumerate(x)}
    networkx.set_node_attributes(G_list_show[k], dict_pos, 'pos')

    dict_v = {i: xi[dim:].tolist() for i, xi in enumerate(x)}
    networkx.set_node_attributes(G_list_show[k], dict_v, attr)

# -----
kwds = kwds_multi_s.copy()

figsize = figsize_multi_s
# -----

out_name = 'anim_some_steps'

show_color_bar = True
same_color_bar = True
title_list = [f't={ddpm.n_steps-t}' for t in t_show]
 
plot_graph_multi_2d_from_G_networkx_list(
        G_list_show, 
        out_name=out_name, 
        nr=None,
        pos_attr='pos',
        attr=attr,
        attr_label_list=attr_label_list, 
        attr_cmap_list=attr_cmap_list,
        title_list=title_list, title_fontsize=12,
        figsize=figsize, save_fig_png=save_fig_png, 
        filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
        with_labels=False, same_color_bar=same_color_bar, show_color_bar=show_color_bar,
        show=plt_show,
        **kwds)        


# In[ ]:


print('Produce animation (2D)...')

# Figure / gif / mp4 using 'imageio' -- Generate images
# =====================================================
out_dir = fig_dir
anim_name = f'{fig_prefix}_{fig_counter:02d}_anim_2d'
fig_counter = fig_counter+1

# ----- Settings --------------
# t_anim_save = np.linspace(0, ddpm.n_steps, 100).astype(int).tolist() 
t_anim_save = np.hstack(
    [np.linspace(                     0, int(0.50*ddpm.n_steps), 10).astype(int),
     np.linspace(int(0.50*ddpm.n_steps), int(0.75*ddpm.n_steps), 20).astype(int)[1:],
     np.linspace(int(0.75*ddpm.n_steps), int(0.90*ddpm.n_steps), 30).astype(int)[1:],
     np.linspace(int(0.90*ddpm.n_steps), ddpm.n_steps          , 50).astype(int)[1:]
    ]).tolist()
# -----------------------------

# Set list of graphs in animation
G_list_anim = [G.copy() for _ in range(len(t_anim_save))]
for k, t in enumerate(t_anim_save):
    x = x_hat_all[t]

    dict_pos = {i: xi[:dim].tolist() for i, xi in enumerate(x)}
    networkx.set_node_attributes(G_list_anim[k], dict_pos, 'pos')

    dict_v = {i: xi[dim:].tolist() for i, xi in enumerate(x)}
    networkx.set_node_attributes(G_list_anim[k], dict_v, attr)

# -----
kwds = kwds_single.copy()

figsize = figsize_single
# -----

if attr is not None:
    # === with attribute(s) ===
    show_color_bar = True
    same_color_bar = True
    
    if same_color_bar:
        # min max of attributes
        v = np.vstack([np.asarray(list(networkx.get_node_attributes(G, attr).values())) for G in G_list_anim])
        vmin_list, vmax_list = v.min(axis=0), v.max(axis=0)
    
    attr_ncomp = len(G_list_anim[0].nodes[0][attr])
    for i_attr in range(attr_ncomp):
        # loop on attributes

        if same_color_bar:
            kwds['vmin'] = vmin_list[i_attr]
            kwds['vmax'] = vmax_list[i_attr]
        else:
            for k in ('vmin', 'vmax'):
                if k in kwds.keys():
                    del kwds[k]

        kwds['cmap'] = attr_cmap_list[i_attr]

        # Generate images (frames)
        # ------------------------
        tmp_dir = os.path.join(out_dir, f'{anim_name}_tmp_{i_attr}')
        frames_path = f'{tmp_dir}/im_{{j:05d}}.png'

        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        print(f'Generate images for animation (png) (attribute {i_attr}) ...')
        fig = plt.figure(figsize=figsize)
        for j, G in enumerate(G_list_anim):
            plot_graph_2d(G, pos_attr='pos', attr=attr, attr_ind=i_attr, with_labels=False, show_colorbar=show_color_bar, **kwds)
            # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            # plt.axis('on')
            plt.axis('equal')
            plt.title(f't={ddpm.n_steps-t_anim_save[j]}')

            plt.savefig(frames_path.format(j=j))

            plt.clf()

        plt.close()

        # Make gif / video (gathering frames)
        # -----------------------------------
        all_frames = [imageio.imread(frames_path.format(j=j)) for j in range(len(G_list_anim))]

        repeat_last_frame = int(len(all_frames)/4)
        nframes = len(all_frames) + repeat_last_frame
        duration = 10 # in seconds
        fps = int(nframes/duration) # frames per second
        #kwargs_gif = {'duration':duration, 'loop':0} # loop = 0 : infinite loop
        kwargs_gif = {'fps':fps, 'loop':0}
        kwargs_mp4 = {'fps':fps}

        # Make gif
        print(f'Generate gif (attribute {i_attr}) ...')
        gif_name = os.path.join(out_dir, f'{anim_name}_{i_attr}.gif')
        imageio.mimsave(gif_name, all_frames + repeat_last_frame*[all_frames[-1]], 'GIF', **kwargs_gif)
        # with imageio.get_writer(gif_name, mode="I", **kwargs) as writer:
        #     for frame in all_frames + repeat_last_frame*[all_frames[-1]]:
        #         writer.append_data(frame)

        # Make video
        print(f'Generate video (attribute {i_attr}) ...')
        mp4_name = os.path.join(out_dir, f'{anim_name}_{i_attr}.mp4')
        imageio.mimsave(mp4_name, all_frames + repeat_last_frame*[all_frames[-1]], 'MP4', **kwargs_mp4)

else:
    # === no attribute ===
    # Generate images (frames)
    # ------------------------
    tmp_dir = os.path.join(out_dir, f'{anim_name}_tmp')
    frames_path = f'{tmp_dir}/im_{{j:05d}}.png'

    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    print(f'Generate images for animation (png) ...')
    fig = plt.figure(figsize=figsize)
    for j, G in enumerate(G_list_anim):
        plot_graph_2d(G, pos_attr='pos', attr=attr, with_labels=False, **kwds)
        # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # plt.axis('on')
        plt.axis('equal')
        plt.title(f't={ddpm.n_steps-t}')

        plt.savefig(frames_path.format(j=j))

        plt.clf()

    plt.close()

    # Make gif / video (gathering frames)
    # -----------------------------------
    all_frames = [imageio.imread(frames_path.format(j=i)) for i in range(len(t_anim_save))]

    repeat_last_frame = int(len(all_frames)/4)
    nframes = len(all_frames) + repeat_last_frame
    duration = 10 # in seconds
    fps = int(nframes/duration) # frames per second
    #kwargs_gif = {'duration':duration, 'loop':0} # loop = 0 : infinite loop
    kwargs_gif = {'fps':fps, 'loop':0}
    kwargs_mp4 = {'fps':fps}

    # Make gif
    print('Generate gif ...')
    gif_name = os.path.join(out_dir, f'{anim_name}.gif')
    imageio.mimsave(gif_name, all_frames + repeat_last_frame*[all_frames[-1]], 'GIF', **kwargs_gif)
    # with imageio.get_writer(gif_name, mode="I", **kwargs) as writer:
    #     for frame in all_frames + repeat_last_frame*[all_frames[-1]]:
    #         writer.append_data(frame)

    # Make video
    print('Generate video ...')
    mp4_name = os.path.join(out_dir, f'{anim_name}.mp4')
    imageio.mimsave(mp4_name, all_frames + repeat_last_frame*[all_frames[-1]], 'MP4', **kwargs_mp4)


# In[ ]:


from IPython import display

# Display gif
if attr is not None:
    i_attr = 0 # choose attribute
    gif_name = os.path.join(out_dir, f'{anim_name}_{i_attr}.gif')
else:
    gif_name = os.path.join(out_dir, f'{anim_name}.gif')

display.Image(gif_name)        


# ### Animation 3D

# In[ ]:


# %%skip_if dim == 2
if dim == 3:
    print('Plot result (3D)...')

    # Plot result - 3D
    # ================
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

    out_name = 'anim_final'
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


# %%skip_if dim == 2
if dim == 3:
    print('Plot result with details (3D)...')

    # Plot result with details
    # ========================

    # ----- Settings --------------
    t_show = np.hstack(
        [np.linspace(                     0, int(0.50*ddpm.n_steps), 6).astype(int),
        np.linspace(int(0.50*ddpm.n_steps), int(0.75*ddpm.n_steps), 7).astype(int)[1:],
        np.linspace(int(0.75*ddpm.n_steps), int(0.90*ddpm.n_steps), 7).astype(int)[1:],
        np.linspace(int(0.90*ddpm.n_steps), ddpm.n_steps          , 7).astype(int)[1:]
        ]).tolist()
    # -----------------------------

    # Set list of graphs to show
    G_list_show = [G.copy() for _ in range(len(t_show))]
    for k, t in enumerate(t_show):
        x = x_hat_all[t]

        dict_pos = {i: xi[:dim].tolist() for i, xi in enumerate(x)}
        networkx.set_node_attributes(G_list_show[k], dict_pos, 'pos')

        dict_v = {i: xi[dim:].tolist() for i, xi in enumerate(x)}
        networkx.set_node_attributes(G_list_show[k], dict_v, attr)

    # -----
    kwargs_edges = kwargs_edges_multi_s.copy()
    kwargs_pts = kwargs_pts_multi_s.copy()
    kwargs_pts_labels = kwargs_pts_labels_multi_s.copy()
    kwargs_scalar_bar = kwargs_scalar_bar_multi_s.copy()

    window_size = window_size_multi_s
    # -----

    notebook = True  # inline
    cpos = None

    out_name = 'anim_some_steps'

    show_color_bar = True
    same_color_bar = True
    title_list = [f't={ddpm.n_steps-t}' for t in t_show]

    plot_graph_multi_3d_from_G_networkx_list(
            G_list_show, 
            out_name=out_name, 
            nr=None,
            pos_attr='pos',
            attr=attr,
            attr_label_list=attr_label_list, 
            attr_cmap_list=attr_cmap_list,
            title_list=title_list, title_fontsize=12,
            notebook=notebook, window_size=window_size, off_screen=False,
            save_fig_png=save_fig_png,
            filename_prefix=f'{fig_dir}/{fig_prefix}_{fig_counter:02d}',
            with_labels=False, same_color_bar=same_color_bar, show_color_bar=show_color_bar,
            kwargs_edges=kwargs_edges, kwargs_pts=kwargs_pts, kwargs_scalar_bar=kwargs_scalar_bar, kwargs_pts_labels=kwargs_pts_labels, 
            cpos=cpos, print_cpos=False)


# In[ ]:


# %%skip_if dim == 2
if dim == 3:
    print('Produce animation (3D)...')

    # Figure / gif / mp4 using 'imageio' -- Generate images
    # =====================================================
    out_dir = fig_dir
    anim_name = f'{fig_prefix}_{fig_counter:02d}_anim_3d'
    fig_counter = fig_counter+1

    # ----- Settings --------------
    # t_anim_save = np.linspace(0, ddpm.n_steps, 100).astype(int).tolist() 
    t_anim_save = np.hstack(
        [np.linspace(                     0, int(0.50*ddpm.n_steps), 10).astype(int),
        np.linspace(int(0.50*ddpm.n_steps), int(0.75*ddpm.n_steps), 20).astype(int)[1:],
        np.linspace(int(0.75*ddpm.n_steps), int(0.90*ddpm.n_steps), 30).astype(int)[1:],
        np.linspace(int(0.90*ddpm.n_steps), ddpm.n_steps          , 50).astype(int)[1:]
        ]).tolist()
    # -----------------------------

    # Set list of graphs in animation
    G_list_anim = [G.copy() for _ in range(len(t_anim_save))]
    for k, t in enumerate(t_anim_save):
        x = x_hat_all[t]

        dict_pos = {i: xi[:dim].tolist() for i, xi in enumerate(x)}
        networkx.set_node_attributes(G_list_anim[k], dict_pos, 'pos')

        dict_v = {i: xi[dim:].tolist() for i, xi in enumerate(x)}
        networkx.set_node_attributes(G_list_anim[k], dict_v, attr)

    # -----
    kwargs_edges = kwargs_edges_single.copy()
    kwargs_pts = kwargs_pts_single.copy()
    kwargs_pts_labels = kwargs_pts_labels_single.copy()
    kwargs_scalar_bar = kwargs_scalar_bar_single.copy()

    window_size = window_size_single
    # -----

    notebook = True  # inline
    cpos = None

    if attr is not None:
        # === with attribute(s) ===
        show_color_bar = True
        same_color_bar = True
        
        if same_color_bar:
            # min max of attributes
            v = np.vstack([np.asarray(list(networkx.get_node_attributes(G, attr).values())) for G in G_list_anim])
            vmin_list, vmax_list = v.min(axis=0), v.max(axis=0)
        
        attr_ncomp = len(G_list_anim[0].nodes[0][attr])
        for i_attr in range(attr_ncomp):
            # loop on attributes

            if same_color_bar:
                kwargs_pts['clim'] = [vmin_list[i_attr], vmax_list[i_attr]]
                kwargs_scalar_bar['title'] = ' ' # same for all
            else:
                if 'clim' in kwargs_pts.keys():
                    del kwargs_pts['clim']
                if 'title' in kwargs_scalar_bar.keys():
                    del kwargs_scalar_bar['title']

            kwargs_pts['cmap'] = attr_cmap_list[i_attr]

            # Generate images (frames)
            # ------------------------
            tmp_dir = os.path.join(out_dir, f'{anim_name}_tmp_{i_attr}')
            frames_path = f'{tmp_dir}/im_{{j:05d}}.png'

            if not os.path.isdir(tmp_dir):
                os.mkdir(tmp_dir)

            print(f'Generate images for animation (png) (attribute {i_attr}) ...')
            fig = plt.figure(figsize=figsize)
            for j, G in enumerate(G_list_anim):
                pp = pv.Plotter(window_size=window_size, off_screen=True)
                kwargs_scalar_bar['title'] = ' '
                plot_graph_3d(
                        G, pos_attr='pos', attr=attr, attr_ind=i_attr, attr_label=attr,
                        plotter=pp, with_labels=False, show_scalar_bar=show_color_bar,
                        kwargs_edges=kwargs_edges, kwargs_pts=kwargs_pts, kwargs_scalar_bar=kwargs_scalar_bar, kwargs_pts_labels=kwargs_pts_labels,
                        cpos=cpos, print_cpos=False)
                pp.add_text(f't={ddpm.n_steps-t_anim_save[j]}', font_size=12)

                # pp.add_bounding_box()
                # pp.show_bounds()
                pp.show_axes()

                pp.screenshot(frames_path.format(j=j))
                pp.close()

            # Make gif / video (gathering frames)
            # -----------------------------------
            all_frames = [imageio.imread(frames_path.format(j=j)) for j in range(len(G_list_anim))]

            repeat_last_frame = int(len(all_frames)/4)
            nframes = len(all_frames) + repeat_last_frame
            duration = 10 # in seconds
            fps = int(nframes/duration) # frames per second
            #kwargs_gif = {'duration':duration, 'loop':0} # loop = 0 : infinite loop
            kwargs_gif = {'fps':fps, 'loop':0}
            kwargs_mp4 = {'fps':fps}

            # Make gif
            print(f'Generate gif (attribute {i_attr}) ...')
            gif_name = os.path.join(out_dir, f'{anim_name}_{i_attr}.gif')
            imageio.mimsave(gif_name, all_frames + repeat_last_frame*[all_frames[-1]], 'GIF', **kwargs_gif)
            # with imageio.get_writer(gif_name, mode="I", **kwargs) as writer:
            #     for frame in all_frames + repeat_last_frame*[all_frames[-1]]:
            #         writer.append_data(frame)

            # Make video
            print(f'Generate video (attribute {i_attr}) ...')
            mp4_name = os.path.join(out_dir, f'{anim_name}_{i_attr}.mp4')
            imageio.mimsave(mp4_name, all_frames + repeat_last_frame*[all_frames[-1]], 'MP4', **kwargs_mp4)

    else:
        # === no attribute ===
        # Generate images (frames)
        # ------------------------
        tmp_dir = os.path.join(out_dir, f'{anim_name}_tmp')
        frames_path = f'{tmp_dir}/im_{{j:05d}}.png'

        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        print(f'Generate images for animation (png) ...')
        fig = plt.figure(figsize=figsize)
        for j, G in enumerate(G_list_anim):
            pp = pv.Plotter(window_size=window_size, off_screen=True)
            kwargs_scalar_bar['title'] = ' '
            plot_graph_3d(
                    G, pos_attr='pos', attr=attr,
                    plotter=pp, with_labels=False,
                    kwargs_edges=kwargs_edges, kwargs_pts=kwargs_pts, kwargs_pts_labels=kwargs_pts_labels,
                    cpos=cpos, print_cpos=False)
            pp.add_text(f't={ddpm.n_steps-t_anim_save[j]}', font_size=12)

            # pp.add_bounding_box()
            # pp.show_bounds()
            pp.show_axes()

            pp.screenshot(frames_path.format(j=j))
            pp.close()

        # Make gif / video (gathering frames)
        # -----------------------------------
        all_frames = [imageio.imread(frames_path.format(j=j)) for j in range(len(G_list_anim))]

        repeat_last_frame = int(len(all_frames)/4)
        nframes = len(all_frames) + repeat_last_frame
        duration = 10 # in seconds
        fps = int(nframes/duration) # frames per second
        #kwargs_gif = {'duration':duration, 'loop':0} # loop = 0 : infinite loop
        kwargs_gif = {'fps':fps, 'loop':0}
        kwargs_mp4 = {'fps':fps}

        # Make gif
        print(f'Generate gif (attribute ...')
        gif_name = os.path.join(out_dir, f'{anim_name}.gif')
        imageio.mimsave(gif_name, all_frames + repeat_last_frame*[all_frames[-1]], 'GIF', **kwargs_gif)
        # with imageio.get_writer(gif_name, mode="I", **kwargs) as writer:
        #     for frame in all_frames + repeat_last_frame*[all_frames[-1]]:
        #         writer.append_data(frame)

        # Make video
        print(f'Generate video (attribute ...')
        mp4_name = os.path.join(out_dir, f'{anim_name}.mp4')
        imageio.mimsave(mp4_name, all_frames + repeat_last_frame*[all_frames[-1]], 'MP4', **kwargs_mp4)


# In[ ]:


from IPython import display

# Display gif
if attr is not None:
    i_attr = 0 # choose attribute
    gif_name = os.path.join(out_dir, f'{anim_name}_{i_attr}.gif')
else:
    gif_name = os.path.join(out_dir, f'{anim_name}.gif')

display.Image(gif_name)        


# ## Karstnet statistics of generated graph

# In[ ]:


print('Compute some Karstnet statistics on generated graph...')

import karstnet as kn

KG = kn.KGraph(G.edges(), networkx.get_node_attributes(G, 'pos'), verbose=True)
KG.characterize_graph()

