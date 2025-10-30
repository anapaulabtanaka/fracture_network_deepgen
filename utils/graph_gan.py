#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File (Python):  'graph_gan.py'
author:         Julien Straubhaar
date:           apr-2024

Functions for Graph GAN.
"""

import numpy as np
import scipy
import networkx

import torch

#  import pyvista as pv
# import matplotlib.pyplot as plt

# # import from package 'geone'
# import geone as gn

# NOTE: load graph_utils.py

# =============================================================================
# Encoding / decoding graph
# =============================================================================
# -----------------------------------------------------------------------------
def encode_graph(G, seq=None, node_attr=None, max_n_nodes=None, max_prev_node=None):
    """
    Encodes a graph and node features (optional).

    Let n_nodes be the number of nodes in the graph. The "encoded adjacency matrix" 
    consists of a 2d arrray `adj_prev_array`, of shape (n_nodes, max_prev_node), defined as 
    
    - `adj_prev_array[i, 0] = 1`,
    - `adj_prev_array[i, j] = 1` if `0 < j <= i` and the node `i` is linked to the node `i-j`,
    - `adj_prev_array[i, j] = 0` otherwise,
    
    i.e., `adj_prev_array[i]` is a sequence of `0` and `1`, starting by `1`, and then 
    specifying for the node `i`, if the previous nodes (up to `max_prev_node`, the maximal 
    number of previous nodes to look back) are connceted (`1`) or not (`0`) to it.

    Moreover, the node features given as attribute of name `node_attr`, if specified, 
    are encoded in a 2d numpy array `node_features_array`, of shape 
    (n_nodes, n_node_features), where `node_features_array[i]` are the features at node i, 
    the "encoded node features".

    Before encoding the nodes of the graph are reorders according to a sequence of 
    visited nodes, `seq`, if specified, i.e. a permutation of [0, ..., n_nodes-1].
        
    This function accounts for at maximum the `max_n_nodes` first nodes of the graph, i.e. 
    truncates the arrays `adj_prev_array` and `node_features_array` after the `max_n_nodes-1` 
    first rows (if needed).

    Parameters
    ----------
    G : networkx.Graph object
        graph, with nodes labels assumed to be integers from 0 (0, 1, 2, ...)
    
    seq : sequence of ints, optional
        permutation of [0, ..., n_nodes-1] where n_nodes is the number of 
        nodes in `G`, sequence of visited nodes, i.e. the "node id" seq[j]
        is the j-th visited node, applied before encoding
        by default (`None`): no permutation, i.e. seq = [0, 1, ..., n_nodes-1] is 
        considered
    
    node_attr : str, optional
        name of the attribute attached to nodes, containing the node features 
        to be encoded; the attribute is a sequence of n_node_features floats; 
        all nodes of the graph are assumed to have this attribute (of same type);
        by default (`None`): node features are not encoded
    
    max_n_nodes : int, optional
        maximal number of nodes in the graph taken into account;
        by default (`None`): `max_n_nodes` is set to n_nodes (the number of nodes 
        in `G`, i.e. all nodes of the graph are taken into account)
    
    max_prev_node : int, optional
        maximal number of nodes to look back;
        by default (`None`): `max_prev_node` is set min(n_nodes, `max_n_nodes`)-1 where 
        n_nodes is the number of nodes in `G`
        
    Returns
    -------
    adj_prev_array : 2d numpy array of shape (n, max_prev_node + 1) of 0 and 1
        encoded ajacency matrix (see above), with n = min(n_nodes, max_n_nodes)
    
    node_features_array : 2d numpy array of shape (n, n_node_features)
        encoded node features (see above)
    """
    if max_n_nodes is None:
        n = G.number_of_nodes()
    else:
        n = min(G.number_of_nodes(), max_n_nodes)

    if max_prev_node is None:
        max_prev_node = n-1
    
    # Initialization of adj_prev_array
    adj_prev_array = np.zeros((n, max_prev_node+1), dtype=int)

    if n == 0:
        if node_attr is not None:
            return adj_prev_array, np.zeros((0, 0))
        else:
            return adj_prev_array

    # Get the adjacency matrix (in csr format)
    adj_mat_csr = networkx.adjacency_matrix(G, seq)

    # Compute each row of adj_prev_array
    adj_prev_array[:n, 0] = 1
    for i in range(1, n):
        jind = adj_mat_csr.indices[adj_mat_csr.indptr[i]:adj_mat_csr.indptr[i+1]]
        jind = jind[np.all((jind < i, jind >= i-max_prev_node), axis=0)]
        adj_prev_array[i, i-jind] = 1

    if node_attr is not None:
        # Get node_features_array
        node_features_array = np.asarray(list(networkx.get_node_attributes(G, node_attr).values())).reshape(G.number_of_nodes(), -1)
        if seq is not None:
            node_features_array = node_features_array[seq]    
        node_features_array = node_features_array[:n]
        
        return adj_prev_array, node_features_array
    else:
        return adj_prev_array
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def decode_graph(adj_prev_array, node_features_array=None, node_attr='feature', get_connected_graph=True):
    """
    Decodes arrays of adjacency and node features ("inverse" of `encode_graph` 
    function).

    Let `adj_prev_array` a 2d numpy array of shape (m, k) an encoded adjacency 
    matrix, and `node_features_array` (if specified) a 2d numpy array of shape 
    (m, n_node_features) the encoded node features (with n_node_features the 
    number of features attached to each node), the result (output) of the 
    function `encode_graph`, i.e. where
    
    - `adj_prev_array` is a binary array (filled with 0 and 1); with
        - `adj_prev_array[i, 0]` : determines the presence (1) of the node i 
        - `adj_prev_array[i, j]`, with `0 < j <= i` : determines the presence (1) \
        of an edge linking node `i` to the node `i-j`,
    - `node_features_array[i, j]` is the feature of index j at node i

    The first index first index i0 s.t. `adj_prev_array[i0, 0] = 0` determines 
    the number of nodes ; if such condition is not reached i0 is set to the 
    number of rows in `adj_prev_array`. Moreover, the number of nodes can be 
    limited to get connected graph (if `get_connected_graph=True`) by considering
    the first index i1, 1 <= i1 < i0 s.t. adj_prev_array[i1, 1:] are all zeros.
    
    This function retrieves the graph (networkx.Graph object) with node features
    (if `node_features_array` specified) set as node attribute with name `node_attr`.

    Parameters
    ----------
    adj_prev_array : 2d numpy array of shape (m, k) of 0 and 1
        encoded adjacency matrix of a graph (see above)
    
    node_features_array : 2d numpy array of shape (m, n_node_features), optional
        encoded node features (see above)
    
    node_attr : str, default: 'feature'
        name of the attribute attached to output graph nodes and containing the 
        node features (used if node_features_array is specified), i.e. 
        node_features_array[i] will be attached to the node i under the name 
        `node_attr` in the output graph
    
    get_connected_graph : bool, default: `True`
        ensure that the decoded graph is connected (see above)
    
    Returns
    -------
    G : networkx.Graph object
        graph (with integers from 0 (0, 1, 2, ...) as nodes labels), decoded from
        `adj_prev_array` and `node_features_array` (see above)
    """
    # Set number of nodes
    # - find the first index i0 s.t. adj_prev_array[i0, 0] = 0
    ind_node_0 = np.where(adj_prev_array[:, 0] == 0)[0]
    if len(ind_node_0):
        n_nodes = ind_node_0[0]
    else:
        n_nodes = adj_prev_array.shape[0]
    
    if get_connected_graph:
        # - limit the number of node to avoid unconnected graph, i.e.
        # find the first index i1, 1 <= i1 < i0 s.t. adj_prev_array[i1, 1:] = 0 for all entries
        ind_node_0 = np.where(adj_prev_array[1:n_nodes, 1:].sum(axis=1) == 0)[0]
    if len(ind_node_0):
        n_nodes = ind_node_0[0] + 1

    # Build the adjacency matrix from `adj_prev_array`
    i_arr, j_arr = np.nonzero(adj_prev_array[:n_nodes, 1:])
    j_arr = i_arr - j_arr - 1
    ind = j_arr >= 0
    i_arr = i_arr[ind]
    j_arr = j_arr[ind]
    i_coo = np.hstack((i_arr, j_arr))
    j_coo = np.hstack((j_arr, i_arr))
    adj_mat_csr = scipy.sparse.coo_array((np.ones(i_coo.shape), (i_coo, j_coo)), shape=(n_nodes, n_nodes)).tocsr()

    # Build the graph
    G = networkx.from_scipy_sparse_array(adj_mat_csr)

    # Attach node features from `node_features_array` (if not None)
    if node_features_array is not None:
        node_features_dict = {i: f for i, f in enumerate(node_features_array)}
        networkx.set_node_attributes(G, node_features_dict, node_attr)

    return G
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Data set to be used with data loader:
#   torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
class Graph_sampler_data_set(object):
    """
    Class defining a data set from a list of graphs.

    The data set delivers (by `__getitem__`) item `(x, n_nodes)` or 
    `(x, x_nf, n_nodes)`, where
    
    - x : 2d tensor of shape `(max_n_nodes, max_prev_node+1)`
        containing the encoded ajacency matrix of a graph (with some nodes 
        numbering)

    - x_nf : 2d tensor of shape `(max_n_nodes, n_node_features)`, optional
        containing the encoded node features (delivered if `node_attr` is 
        specified)

    - n_nodes : int
        number of nodes taking into account in the encoding of the graph
        (`n_nodes <= max_n_nodes`), i.e.:
        - the first `n_nodes-1` rows of `x` (resp. `x_nf`) contain the encoded \
        adjacency matrix (resp. encoded node features)
        - the next rows of `x` and `x_nf` are filled with zeros

    The encoded information is retrieved using the function `encode_graph`.

    Parameters `max_n_nodes` and `max_prev_node` are set by the constructor
    and can be automatically computed (see `__init__`). The number of features 
    attached to each node, `n_node_features`, is automatically determined 
    (from the graph(s) given to the constructor) and name `node_attr`; all nodes 
    of each graph are assumed to have an attribute of same name (`node_attr`) which 
    is a sequence of length `n_node_features`; if `node_attr` is not specified, 
    n_nodes_feature is equal to 0 and x_nf is not delivered.
    

    Notes
    -----
    - if `max_prev_node` is not specified, it is computed by the method \
    `calc_max_prev_node`, whose keyword arguments (dict.) can be passed  \
    through the parameter `calc_max_prev_node_kwargs`; for reproducibility, a \
    `seed` can be specified in `calc_max_prev_node_kwargs`
    - methods `__len__` and `__getitem__` must be defined, so that instanciated \ 
    data set can be used with data loader from `pytorch` \
    (`torch.utils.data.DataLoader`)
    - before using a data loader, use `torch.random.manual_seed()` to ensure \
    reproducibility of batches delivered by the data loader (if needed)

    """
    def __init__(self, G_list, G_nsample, node_attr=None, 
                 use_bfs=True, max_n_nodes=None, max_prev_node=None, 
                 node_features_noise_std=None, calc_max_prev_node_kwargs=None):
        """
        Initializes the class.

        Parameters
        ----------
        G_list : list of `networkx.Graph`
            list of graph
        
        G_nsample : sequence of ints (>=1)
            sequence of same length as `G_list`, of ints >= 1, 
            number of times that each graph in `G_list` is sampled, i.e.
            G_list[i] will be sampled G_nsample[i] times;
            hence, the length of "data set" is the cumulative sum of `G_nsample`
        
        node_attr : str, optional
            name of the attribute attached to nodes, the attribute is a sequence
            of n_node_features floats; all nodes of every graph in `G_list` are 
            assumed to have this attribute (of same type);
            by default (`None`): node features are not considered
        
        use_bfs : bool, default: `True`
            - if `True`: BFS (breadth-first-search) is used before encoding 
            - if `False`: BFS is not used before encoding
        
        max_n_nodes : int, optional
            maximal number of nodes taken into account (in a graph);
            by default (`None`): `max_n_nodes` is set to the maximum of the number 
            of nodes of graphs in `G_list`
        
        max_prev_node : int, optional
            maximal number of previous nodes used for encoding adjacency matrices;
            by default (`None`): `max_prev_node` is calculated using the method
            `calc_max_prev_node`
        
        node_features_noise_std : float or sequence of positive values, optional
            if specified: float or sequence of length n_node_features, standard 
            deviation of gaussian random noise, that will be added to node features
        
        calc_max_prev_node_kwargs : dict, optional
            keyword arguments to be passed to method `calc_max_prev_node` (used if
            `max_prev_node=None`)
        """
        # List of graphs 
        self.G_list = G_list

        # List of number of nodes of each graph
        self.G_n_nodes_list = [G.number_of_nodes() for G in G_list]

        # List of indices of sampled graph
        # - G_nsample[i] times i, for i 0, 1, ... len(G_list)-1
        self.G_index_list = np.repeat(range(len(G_nsample)), G_nsample)

        # Attribute to be considered
        self.node_attr = node_attr

        # Number of features (in the considered attribute)
        if self.node_attr is not None:
            self.n_node_features = len(self.G_list[0].nodes[0][self.attr])
            # note: all nodes of every graph in `G_list` are  assumed to have attribute attr (of same type)
        else:
            self.n_node_features = 0

        # node_features_noise_std: should be None, float, or a a numpy array of shape (self.n_node_features, )
        self.node_features_noise_std = node_features_noise_std
                
        # Length of the data set
        self.len = len(self.G_index_list)

        # Use BFS sequence
        self.use_bfs = use_bfs

        if max_n_nodes is None:
            self.max_n_nodes = max(self.G_n_nodes_list)
        else:
            self.max_n_nodes = max_n_nodes

        if max_prev_node is None:
            if calc_max_prev_node_kwargs is None:
                calc_max_prev_node_kwargs = {}
            self.max_prev_node = self.calc_max_prev_node(**calc_max_prev_node_kwargs)
        else:
            self.max_prev_node = max_prev_node 

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Select the graph 
        G_ind = self.G_index_list[idx]
        G = self.G_list[G_ind].copy()
        
        # Encodes graph (starting by reordering nodes randomly)
        # - use only random generator from torch -> reproducibility is then
        #   guaranteed by setting `torch.random.manual_seed()` 
        # seq = np.random.permutation(G.number_of_nodes()) # based on numpy
        seq = torch.randperm(G.number_of_nodes()).numpy()
        if self.use_bfs:
            adj_mat_csr = networkx.adjacency_matrix(G, seq)
            if self.node_attr is not None:
                node_features_array = np.array(list(networkx.get_node_attributes(G, self.node_attr).values()))[seq]
            G = networkx.from_scipy_sparse_array(adj_mat_csr)
            if self.node_attr is not None:
                node_features_dict = {i: f for i, f in enumerate(node_features_array)}
                networkx.set_node_attributes(G, node_features_dict, self.node_attr)

            seq = get_bfs_sequence(G, 0)
        
        if self.node_attr is not None:
            # Encodes graph with node features
            adj_prev_array, node_features_array = encode_graph(G, seq, node_attr=self.node_attr, max_n_nodes=self.max_n_nodes, max_prev_node=self.max_prev_node)
        else:
            # Encodes graph without node features
            adj_prev_array = encode_graph(G, seq, node_attr=self.node_attr, max_n_nodes=self.max_n_nodes, max_prev_node=self.max_prev_node)

        # -> adj_prev_array      of shape (n, self.max_prev_node + 1), with n <= self.max_n_nodes
        # -> node_features_array of shape (n, self.n_node_features) [optional]


        # Define x, x_nf and n_nodes (to be delivered as item) [x_nf: optional]
        # x from adj_prev_array
        x = torch.zeros((self.max_n_nodes, self.max_prev_node+1))#, dtype=torch.float32)
        x[0:adj_prev_array.shape[0], :] = torch.from_numpy(adj_prev_array)

        # n_nodes: number of nodes taken into account in the encoding
        n_nodes = adj_prev_array.shape[0]

        if self.node_attr is not None:
            # x_nf from node_features_array
            x_nf = torch.zeros((self.max_n_nodes, self.n_node_features))#, dtype=torch.float32)
            x_nf[0:node_features_array.shape[0], :] = torch.from_numpy(node_features_array)

            return x, x_nf, n_nodes
        else:
            return x, n_nodes

    def calc_max_prev_node(self, nsample=10000, quantile=0.95, seed=None, verbose=1):
        """
        Computes a value for `max_prev_node`. 
        
        This function computes the bandwidth (computed on first 
        `self.max_n_nodes` nodes at maximum) of `nsample` graphs 
        sampled from the list `self.G_list`, retrieves the quantile
        `quantile` q of all bandwidths, and returns `int(q)` as value 
        for `self.max_prev_node`.
        Note that the bandwidth of a matrix :math:`M=(m_{ij})` is defined as 
        :math:`bw = \max\{|i-j| : m_{ij} \\neq 0\}`
        (i.e. a diagonal matrix has a bandwidth of 0 with this definition).
        
        Parameters
        ----------
        nsample : int, default: 20000
            number of sampled graphs
        
        quantile : float, default: 0.95
            quantile to compute for all bandwidths of sampled graphs
            (see above)
        
        seed : int, optional
            seed for initializing random number generator (`numpy`)
        
        verbose : int, default: 1
            - if 0: do not show progress
            - if 1: show (print) progress

        Returns
        -------
        max_prev_node : int
            calculated `max_prev_node` (see above)

        Notes 
        -----
        `self.G_list`, `self.G_index_list`, `self.use_bfs`, `self.max_n_nodes` 
        and `self.len` must be defined.
        """
        if seed is not None:
            np.random.seed(seed)
        if verbose:
            progress_old, progress = -1, 0
        bw = np.zeros(nsample)
        for i in range(nsample):
            if verbose:
                progress = int(100*(i+1)/nsample)
                if progress > progress_old:
                    print(f'Compute max_prev_node {progress}%...')
                    progress_old = progress
            G_ind = self.G_index_list[np.random.randint(self.len)]
            G = self.G_list[G_ind].copy()
            seq = np.random.permutation(G.number_of_nodes())
            adj_mat_csr = networkx.adjacency_matrix(G, seq)
            if self.use_bfs:
                G = networkx.from_scipy_sparse_array(adj_mat_csr)
                seq = get_bfs_sequence(G, 0)
                adj_mat_csr = networkx.adjacency_matrix(G, seq)
            n_nodes = min(G.number_of_nodes(), self.max_n_nodes)
            bw[i] = min(
                np.max([np.max(np.abs(i - adj_mat_csr.indices[adj_mat_csr.indptr[i]:adj_mat_csr.indptr[i+1]])) for i in range(n_nodes)]),
                n_nodes-1)

        max_prev_node = int(np.quantile(bw, q=quantile))
        return max_prev_node
# -----------------------------------------------------------------------------

# =============================================================================
# GAN model for graph generation (not accounting for node features)
# =============================================================================
# ------------------------------------------------------------------------------
# Gan design: Generator / Discriminator
class Generator_mlp(torch.nn.Module):
    """
    Generator.
    
    It is composed of several MLP layers. No activation is done after the last
    layer.
    """
    def __init__(self, latent_dim, max_n_nodes, max_prev_node, mlp_hidden_dims, batch_norm=False, dropout=0.0, activation='ReLU'):
        """
        Initializes the class.

        Parameters
        ----------
        latent_dim : int
            dimension of the latent space
        
        max_n_nodes, max_prev_node : int
            maximal number of nodes, and maximal of number of previous nodes
            when encoding graph;
            output size corresponds to max_n_nodes*(max_prev_node+1)
        
        mlp_hidden_dims: list
            list of dimensions of the MLP hidden layers
        
        batch_norm : bool, defaul: False
            if `True`: Batch normalization is done after each MLP layer except 
            the last one
        
        dropout : float, default: 0.0
            float in [0, 1), if > 0, probability of Dropout, applied after each 
            MLP layer(+Batch normalization) except the last one
        
        activation : str, {'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'}
            non-linearity activation after each MLP layers execpt the last one
        """
        super().__init__()        
        self.latent_dim = latent_dim
        self.max_n_nodes = max_n_nodes
        self.max_prev_node = max_prev_node
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        if activation == 'ReLU': 
            self.activation_fun = torch.nn.ReLU
        elif activation == 'LeakyReLU': 
            self.activation_fun = torch.nn.LeakyReLU # default negative slope
        elif activation == 'Sigmoid': 
            self.activation_fun = torch.nn.Sigmoid
        elif activation == 'Tanh': 
            self.activation_fun = torch.nn.Tanh

        mlp_hidden_layers = []
        for in_dim, out_dim in zip([self.latent_dim]+self.mlp_hidden_dims[:-1], self.mlp_hidden_dims):
            mlp_hidden_layers.append(torch.nn.Linear(in_dim, out_dim))
            mlp_hidden_layers.append(self.activation_fun())
            if batch_norm:
                mlp_hidden_layers.append(torch.nn.BatchNorm1d(out_dim))
            if self.dropout > 0:
                mlp_hidden_layers.append(torch.nn.Dropout(p=dropout))
        
        self.mlp_hidden_layers = torch.nn.Sequential(*mlp_hidden_layers)
        
        self.mlp_out = torch.nn.Linear(self.mlp_hidden_dims[-1], self.max_n_nodes*(self.max_prev_node+1))
        
        self.init_weights()

    def forward(self, x):
        output = self.mlp_hidden_layers(x)
        output = self.mlp_out(output)
        output = output.view(-1, self.max_n_nodes, self.max_prev_node+1)
        return output

    def init_weights(self, gain=1.0, seed=None):
        """Initializes weights of the network."""
        if seed is not None:
            torch.random.manual_seed(seed)

        for name, param in self.named_parameters():
            # print('...', name, param.ndim, param.size())
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name and param.ndim > 1:
                torch.nn.init.xavier_uniform_(param, gain=gain)
                                                    # gain=nn.init.calculate_gain('sigmoid')
                                                    # gain=nn.init.calculate_gain('relu')
                                                    # ...
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class Discriminator_mlp(torch.nn.Module):
    """
    Discriminator.
    
    It is composed of several MLP layers. No activation is done after the last
    layer.
    """
    def __init__(self, max_n_nodes, max_prev_node, mlp_hidden_dims, batch_norm=False, dropout=0.0, activation='ReLU'):
        """
        Initializes the class.

        Parameters
        ----------
        latent_dim : int
            dimension of the latent space
        
        max_n_nodes, max_prev_node : int
            maximal number of nodes, and maximal of number of previous nodes
            when encoding graph;
            output size corresponds to max_n_nodes*(max_prev_node+1)
        
        mlp_hidden_dims: list
            list of dimensions of the MLP hidden layers
        
        batch_norm : bool, defaul: False
            if `True`: Batch normalization is done after each MLP layer except 
            the last one
        
        dropout : float, default: 0.0
            float in [0, 1), if > 0, probability of Dropout, applied after each 
            MLP layer(+Batch normalization) except the last one
        
        activation : str, {'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh'}
            non-linearity activation after each MLP layers execpt the last one
        """
        super().__init__()        
        self.max_n_nodes = max_n_nodes
        self.max_prev_node = max_prev_node
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        if activation == 'ReLU': 
            self.activation_fun = torch.nn.ReLU
        elif activation == 'LeakyReLU': 
            self.activation_fun = torch.nn.LeakyReLU # default negative slope
        elif activation == 'Sigmoid': 
            self.activation_fun = torch.nn.Sigmoid
        elif activation == 'Tanh': 
            self.activation_fun = torch.nn.Tanh

        mlp_hidden_layers = []
        for in_dim, out_dim in zip([self.max_n_nodes*(self.max_prev_node+1)]+self.mlp_hidden_dims[:-1], self.mlp_hidden_dims):
            mlp_hidden_layers.append(torch.nn.Linear(in_dim, out_dim))
            mlp_hidden_layers.append(self.activation_fun())
            if batch_norm:
                mlp_hidden_layers.append(torch.nn.BatchNorm1d(out_dim))
            if self.dropout > 0:
                mlp_hidden_layers.append(torch.nn.Dropout(p=dropout))
        
        self.mlp_hidden_layers = torch.nn.Sequential(*mlp_hidden_layers)
        
        self.mlp_out = torch.nn.Linear(self.mlp_hidden_dims[-1], 1)
        
        self.init_weights()

    def forward(self, x):
        output = torch.flatten(x, start_dim=1) # OR: # output = torch.nn.Flatten()(x)
        output = self.mlp_hidden_layers(output)
        output = self.mlp_out(output)
        return output

    def init_weights(self, gain=1.0, seed=None):
        """Initializes weights of the network."""
        if seed is not None:
            torch.random.manual_seed(seed)

        for name, param in self.named_parameters():
            # print('...', name, param.ndim, param.size())
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name and param.ndim > 1:
                torch.nn.init.xavier_uniform_(param, gain=gain)
                                                    # gain=nn.init.calculate_gain('sigmoid')
                                                    # gain=nn.init.calculate_gain('relu')
                                                    # ...
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Gan design: Generator / Discriminator
class Generator_conv(torch.nn.Module):
    """
    Generator.
    """
    def __init__(self, latent_dim, max_n_nodes, max_prev_node, 
                 n_channels, kernel_stride_padding_params_dim0, kernel_stride_padding_params_dim1, 
                 batch_norm=False, dropout=0.0, activation='ReLU'):
        """
        """
        super().__init__()        
        self.latent_dim = latent_dim
        self.max_n_nodes = max_n_nodes
        self.max_prev_node = max_prev_node
        self.n_channels = n_channels
        self.kernel_stride_padding_params_dim0 = kernel_stride_padding_params_dim0
        self.kernel_stride_padding_params_dim1 = kernel_stride_padding_params_dim1
        self.dropout = dropout

        if activation == 'ReLU': 
            self.activation_fun = torch.nn.ReLU
        elif activation == 'LeakyReLU': 
            self.activation_fun = torch.nn.LeakyReLU # default negative slope
        elif activation == 'Sigmoid': 
            self.activation_fun = torch.nn.Sigmoid
        elif activation == 'Tanh': 
            self.activation_fun = torch.nn.Tanh

        n_channels_ext = [latent_dim] + n_channels
        last_layer = np.zeros(len(kernel_stride_padding_params_dim1), dtype='bool')
        last_layer[-1] = True

        conv_layers = []
        for c0, c1, (k0, s0, p0), (k1, s1, p1), ll in zip(n_channels_ext[:-1], n_channels_ext[1:], self.kernel_stride_padding_params_dim0, self.kernel_stride_padding_params_dim1, last_layer):
            conv_layers.append(torch.nn.ConvTranspose2d(c0, c1, (k0, k1), (s0, s1), (p0, p1)))
            if ll:
                break
            conv_layers.append(self.activation_fun())
            if batch_norm:
                conv_layers.append(torch.nn.BatchNorm2d(c1))
            if self.dropout > 0:
                conv_layers.append(torch.nn.Dropout(p=dropout))
        
        self.conv_layers = torch.nn.Sequential(*conv_layers)
        
        self.init_weights()

    def forward(self, x):
        output = self.conv_layers(x.view(-1, self.latent_dim, 1, 1))
        output = output.view(-1, self.max_n_nodes, self.max_prev_node+1) # n_channels[-1] should be 1
        return output

    def init_weights(self, gain=1.0, seed=None):
        """Initializes weights of the network."""
        if seed is not None:
            torch.random.manual_seed(seed)

        for name, param in self.named_parameters():
            # print('...', name, param.ndim, param.size())
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name and param.ndim == 4:
                c = gain * 1.0/np.sqrt(param.size(0)*param.size(2)*param.size(3))
                torch.nn.init.uniform_(param, a=-c, b=c)
                # torch.nn.init.xavier_uniform_(param, gain=gain)
                                                    # gain=nn.init.calculate_gain('sigmoid')
                                                    # gain=nn.init.calculate_gain('relu')
                                                    # ...
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class Discriminator_conv(torch.nn.Module):
    """
    Discriminator.
    """
    def __init__(self, max_n_nodes, max_prev_node, 
                 n_channels, kernel_stride_padding_params_dim0, kernel_stride_padding_params_dim1, 
                 batch_norm=False, dropout=0.0, activation='ReLU'):
        """
        """
        super().__init__()        
        self.max_n_nodes = max_n_nodes
        self.max_prev_node = max_prev_node
        self.n_channels = n_channels
        self.kernel_stride_padding_params_dim0 = kernel_stride_padding_params_dim0
        self.kernel_stride_padding_params_dim1 = kernel_stride_padding_params_dim1
        self.dropout = dropout

        if activation == 'ReLU': 
            self.activation_fun = torch.nn.ReLU
        elif activation == 'LeakyReLU': 
            self.activation_fun = torch.nn.LeakyReLU # default negative slope
        elif activation == 'Sigmoid': 
            self.activation_fun = torch.nn.Sigmoid
        elif activation == 'Tanh': 
            self.activation_fun = torch.nn.Tanh

        n_channels_ext = n_channels + [1]
        last_layer = np.zeros(len(kernel_stride_padding_params_dim1), dtype='bool')
        last_layer[-1] = True

        conv_layers = []
        for c0, c1, (k0, s0, p0), (k1, s1, p1), ll in zip(n_channels_ext[:-1], n_channels_ext[1:], self.kernel_stride_padding_params_dim0, self.kernel_stride_padding_params_dim1, last_layer):
            conv_layers.append(torch.nn.Conv2d(c0, c1, (k0, k1), (s0, s1), (p0, p1)))
            if ll:
                break
            conv_layers.append(self.activation_fun())
            if batch_norm:
                conv_layers.append(torch.nn.BatchNorm2d(c1))
            if self.dropout > 0:
                conv_layers.append(torch.nn.Dropout(p=dropout))
        
        self.conv_layers = torch.nn.Sequential(*conv_layers)
        
        self.init_weights()

    def forward(self, x):
        output = self.conv_layers(x.view(-1, self.n_channels[0], self.max_n_nodes, self.max_prev_node+1))
        output = torch.nn.Flatten()(output)
        return output

    def init_weights(self, gain=1.0, seed=None):
        """Initializes weights of the network."""
        if seed is not None:
            torch.random.manual_seed(seed)

        for name, param in self.named_parameters():
            # print('...', name, param.ndim, param.size())
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name and param.ndim == 4:
                c = gain * 1.0/np.sqrt(param.size(0)*param.size(2)*param.size(3))
                torch.nn.init.uniform_(param, a=-c, b=c)
                # torch.nn.init.xavier_uniform_(param, gain=gain)
                                                    # gain=nn.init.calculate_gain('sigmoid')
                                                    # gain=nn.init.calculate_gain('relu')
                                                    # ...
# ------------------------------------------------------------------------------

# ===== GAN (classic) =====
# ------------------------------------------------------------------------------
def train_gan(
        model_D, 
        model_G,
        optimizer_D, 
        optimizer_G, 
        data_loader,
        lr_scheduler_D=None,
        lr_scheduler_G=None,
        return_lr=False,
        return_loss=True,
        return_accuracy=False,
        return_score=False,
        num_epochs=10,
        print_epoch=1,
        save_fake_epoch=0,
        z_fixed=None,
        save_fake_file_fmt='./fake_{:04d}.pt',
        device=torch.device('cpu')):
    """
    Trains a GAN model.

    Train the discriminator, then the generator (using new fake data).

    Parameters
    ----------
    model_D : network
        model (network) for the discriminator
    
    model_G : network
        model (network) for the generator
    
    optimizer_D, optimizer_G : optimizer
        updater, torch optimizer for network model_D, model_G (resp.), e.g. 
        `torch.optim.SGD(net.parameters(), weight_decay=0.001, lr=0.03)`
    
    data_loader : data loader
        yielding mini-batches (x, n_nodes) from a data set, with:

            - B: current batch size
            - N: data_set.max_n_nodes
            - P: data_set.max_prev_node

            - x: tensor of shape (B, N, P+1)
                x[k]: k-th input of the mini-batch (encoded adjacency matrix)
            - n_nodes: tensor of shape (B, )
                n_nodes[k]: number of nodes taken into account in the encoding
                `x[k]`

    lr_scheduler_D, lr_scheduler_G : scheduler, optional
        learning rate scheduler for network model_D, model_G (resp.), e.g. 
        `torch.optim.lr_scheduler.*`
    
    return_lr : bool, default: `False`
        if `True`: return two sequences of learning rates used:
        - dicriminator lr
        - generator lr
    
    return_loss : bool, default: `True`
        if `True`: return four sequences of losses:
        - discriminator loss for real data
        - discriminator loss for fake data
        - discriminator loss (mean)
        - generator loss
    
    return_loss : bool, default: `True`
        if `True`: return losses (see below)
    
    return_accuracy : bool, default: `False`
        if `True`: return three sequences of accuracies:
        
        - accuracy of the discriminator for real data
        - accuracy of the discriminator for fake data
        - accuracy of the discriminator (mean of accuracy for real and \
        fake data);
        
        accuracy is defined as the rate of true prediction (real or fake)
    
    return_score : bool, default: `False`
        if `True`: return three sequences of scores:

        - score of the discriminator for real data
        - score of the discriminator for fake data
        - score of the discriminator (mean of score for real and \
        fake data);

        score is defined as the mean score over single input; if p is the
        predicted probability to be real for a single input, the corresponding
        score is s=p for real input and s=(1-p) for fake input

    num_epochs : int, default: 10
        number of epochs
    
    print_epoch : int, default: 10
        result of every `print_epoch` epoch is displayed in stdout, if 
        `print_epoch > 0`
    
    save_fake_epoch : int, default: 0
        if `save_fake_epoch > 0`, at each generated fake data, using z_fixed as input and 
        a tensor of generated fake data, `model_G(z_fixed)`, at every `save_fake_epoch` epoch 
        is saved (written on the disk), provided `z_fixed` is specified (not `None`)
    
    z_fixed : tensor of 2 dimensions, optional
        tensor of random noise in N(0,1), used to generate fake data to save (see 
        `save_fake_epoch`), its shape should be (m, latent_dim) where m is the number of 
        fake data to be generated and latent_dim the dimension of the latent space; 
        unused if `save_fake_epoch<=0`
    
    save_fake_file_fmt : str, default: './fake_{:04d}.pt'
        string for filename (including path) for saving generated fake data 
        (see `save_fake_epoch`), at epoch i the file will be `save_fake_file_fmt.format(i)`
    
    device : torch device, default: torch.device('cpu')
        device on which the model is trained
    
    Returns
    -------
    loss_D_real : list, optional
        returned if `return_loss=True`, discriminator loss over real data 
        of every epoch, list of floats of length `num_epochs`
    
    loss_D_fake : list, optional
        returned if `return_loss=True`, discriminator loss over fake data 
        of every epoch, list of floats of length `num_epochs`
    
    loss_D : list, optional
        returned if `return_loss=True`, discriminator loss of every epoch, 
        list of floats of length `num_epochs`
    
    loss_G : list, optional
        returned if `return_loss=True`, generator loss of every epoch, 
        list of floats of length `num_epochs`
    
    accuracy_real : list, optional
        returned if `return_accuracy=True`, accuracy of the discriminator
        for real data (number of true prediction over total number of 
        predictions) of every epoch, 
        list of floats of length `num_epochs`
    
    accuracy_fake : list, optional
        returned if `return_accuracy=True`, accuracy of the discriminator
        for fake, i.e. generated, data (number of true prediction over total 
        number of predictions) of every epoch, 
        list of floats of length `num_epochs`
    
    accuracy : list, optional
        returned if `return_accuracy=True`, accuracy of the discriminator,
        mean of `accuracy_real` and `acccuracy_fake`;
        list of floats of length `num_epochs`
    
    score_real : list, optional
        returned if `return_score=True`, score of the discriminator
        for real data (mean of predicted probabilities);
        list of floats of length `num_epochs`
    
    score_fake : list, optional
        returned if `return_score=True`, score of the discriminator
        for fake data (one minus mean of predicted probabilities);
        list of floats of length `num_epochs`
    
    score : list, optional
        returned if `return_score=True`, score of the discriminator,
        mean of `score_real` and `score_fake`;
        list of floats of length `num_epochs`
    
    lr_used_D : list, optional
        returned if `return_lr=True`, learning rate for the discriminator
        used at each epoch, list of floats of length `num_epochs`,        
    
    lr_used_G : list, optional
        returned if `return_lr=True`, learning rate for the generator
        used at each epoch, list of floats of length `num_epochs`,        
    
    Notes
    -----
    With n the total number of members in the training set
    the parameters batch_size and update_D_n_steps should satified:
    
    - n = m * batch_size (i.e. n multiple of batch_size)
    - m = k * update_D_n_steps (i.e. m multiple of update_D_n_steps)
    """
    fname = 'train_gan'

    # Copy networks to device
    model_D.to(device)
    model_G.to(device)

    model_D.train() # set the Discriminator network in training mode
    model_G.train() # set the Generator     network in training mode

    # Dimension of latent space
    latent_dim = model_G.latent_dim

    # Initialize list for loss
    if return_loss:
        loss_D_real, loss_D_fake, loss_G = [], [], []
        loss_D_real_epoch, loss_D_fake_epoch, loss_G_epoch = 0.0, 0.0, 0.0 # reset loss of one epoch

    # Initialize list for accuracy
    if return_accuracy:
        accuracy_real, accuracy_fake = [], []
        accuracy_real_epoch, accuracy_fake_epoch = 0.0, 0.0 # reset accuracy of one epoch

    # Initialize list for score
    if return_score:
        score_real, score_fake = [], []
        score_real_epoch, score_fake_epoch = 0.0, 0.0 # reset score of one epoch

    # Initialize list for lr
    if return_lr:
        lr_used_D = []
        lr_used_G = []

    # Check save_fake_epoch and z_fixed
    if save_fake_epoch:
        if z_fixed is None or z_fixed.ndim != 2 or z_fixed.shape[1] != latent_dim:
            print(f'ERROR ({fname}): `z_fixed` not valid')
            return
        z_fixed = z_fixed.to(device)

    # Train the network
    for epoch in range(num_epochs):
        # Train GAN through every mini-batch
        train_len = 0  # reset train length
        for d in data_loader:
            X = d[0]
            # n_nodes = d[1]
            
            # one mini-batch
            X = X.to(device) # copy to device

            # Create tensor of labels (real and fake)
            real_label = torch.full((len(X), 1), 1, dtype=X.dtype, device=device)
            fake_label = torch.full((len(X), 1), 0, dtype=X.dtype, device=device)
            
            # Train the Discriminator
            # -----------------------
            X_real_out = model_D(X) # Discriminator forward for real data            
            X_real_out = torch.sigmoid(X_real_out) # OR: X_real_out = torch.nn.Sigmoid()(X_real_out)

            # Generate fake data via the Generator
            z = torch.randn(len(X), latent_dim, device=device)
            X_fake = model_G(z) # fake data via Generator forward
            X_fake = torch.sigmoid(X_fake) # OR: X_fake = torch.nn.Sigmoid()(X_fake)
            X_fake_out = model_D(X_fake.detach()) # Discriminator forward for fake data
                                                  # - Use detach() to not track the gradient
                                                  # because the Discriminator is trained!
            X_fake_out = torch.sigmoid(X_fake_out)

            # Compute Discriminator loss for the current mini-batch (mb)
            loss_D_real_mb = torch.nn.functional.binary_cross_entropy(X_real_out, real_label) # loss for real data
            loss_D_fake_mb = torch.nn.functional.binary_cross_entropy(X_fake_out, fake_label) # loss for fake data 
            # loss_D_real_mb = torch.nn.BCELoss()(X_real_out, real_label) # loss for real data
            # loss_D_fake_mb = torch.nn.BCELoss()(X_fake_out, fake_label) # loss for fake data 
            loss_D_mb = 0.5*(loss_D_real_mb + loss_D_fake_mb)

            # Update Discriminator parameters
            optimizer_D.zero_grad() # reset gradient in optimized tensors of Discriminator
            loss_D_mb.backward()    # compute gradient of Discriminator loss wrt to Discriminator parameters
            optimizer_D.step()      # one step in Discriminator training (update)

            # Train the Generator
            # -------------------
            # Generate fake data via the Generator
            # (or re-use same fake data: comment the three following lines)
            z = torch.randn(len(X), latent_dim, device=device) 
            X_fake = model_G(z) # fake data via Generator forward
            X_fake = torch.sigmoid(X_fake) # OR: X_fake = torch.nn.Sigmoid()(X_fake)
            X_fake_out = model_D(X_fake) # Discriminator forward for fake data
            X_fake_out = torch.sigmoid(X_fake_out)

            # Compute Generator loss for the current mini-batch (mb) by giving fake data and real label
            loss_G_mb = torch.nn.functional.binary_cross_entropy(X_fake_out, real_label)
            # loss_G_mb = torch.nn.BCELoss()(X_fake_out, real_label)
            
            # Update Generator parameters
            optimizer_G.zero_grad() # reset gradient in optimized tensors of Generator
            loss_G_mb.backward()    # compute gradient of Generator loss wrt to Generator parameters
            optimizer_G.step()      # one step in Generator training (update)

            if return_loss:
                with torch.no_grad():
                    loss_D_real_epoch += loss_D_real_mb.item()*X.shape[0]
                    loss_D_fake_epoch += loss_D_fake_mb.item()*X.shape[0]
                    loss_G_epoch += loss_G_mb.item()*X.shape[0]

            if return_accuracy:
                with torch.no_grad():
                    accuracy_real_epoch += (X_real_out.view(-1) >  0.5).sum().item()
                    accuracy_fake_epoch += (X_fake_out.view(-1) <= 0.5).sum().item()

            if return_score:
                with torch.no_grad():
                    score_real_epoch += X_real_out.sum().item()
                    score_fake_epoch += X_fake_out.sum().item()

            train_len += X.shape[0]


        if return_lr:
            lr_used_D.append(optimizer_D.param_groups[0]['lr'])
            lr_used_G.append(optimizer_G.param_groups[0]['lr'])
        if lr_scheduler_D:
            # lr_used_D.append(lr_scheduler_D.get_last_lr()[0])
            lr_scheduler_D.step()
        if lr_scheduler_G:
            # lr_used_G.append(lr_scheduler_G.get_last_lr()[0])
            lr_scheduler_G.step()

        if return_loss:
            loss_D_real.append(loss_D_real_epoch/train_len)
            loss_D_fake.append(loss_D_fake_epoch/train_len)
            loss_G.append(loss_G_epoch/train_len)
            loss_D_real_epoch, loss_D_fake_epoch, loss_G_epoch = 0.0, 0.0, 0.0 # reset loss of one epoch

        if return_accuracy:
            accuracy_real.append(accuracy_real_epoch/train_len)
            accuracy_fake.append(accuracy_fake_epoch/train_len)
            accuracy_real_epoch, accuracy_fake_epoch = 0.0, 0.0 # reset accuracy of one epoch

        if return_score:
            score_real.append(score_real_epoch/train_len)
            score_fake.append(1.0 - score_fake_epoch/train_len)
            score_real_epoch, score_fake_epoch = 0.0, 0.0 # reset accuracy of one epoch

        if print_epoch > 0 and epoch % print_epoch == 0:
            # Print result of current epoch
            s = f'epoch {epoch+1} of {num_epochs}'
            if return_loss:
                s = s + f', loss_G: {loss_G[-1]:.4f}, loss_D (real, fake, mean): {loss_D_real[-1]:.4f}, {loss_D_fake[-1]:.4f}, {0.5*(loss_D_real[-1]+loss_D_fake[-1]):.4f}'
            if return_accuracy:
                s = s + f', accuracy: {accuracy_real[-1]:.4f}, {accuracy_fake[-1]:.4f}, {0.5*(accuracy_real[-1]+accuracy_fake[-1]):.4f}'
            if return_score:
                s = s + f', score: {score_real[-1]:.4f}, {score_fake[-1]:.4f}, {0.5*(score_real[-1]+score_fake[-1]):.4f}'
            print(s)

        if save_fake_epoch > 0 and epoch%save_fake_epoch == 0:
            with torch.no_grad():
                X_fake_save = model_G(z_fixed)
                X_fake_save = torch.sigmoid(X_fake_save).to('cpu')
                torch.save(X_fake_save, save_fake_file_fmt.format(epoch))

    # Set networks on cpu
    model_D.to(torch.device('cpu'))
    model_G.to(torch.device('cpu'))

    out = []
    if return_loss:
        out.append(loss_D_real)
        out.append(loss_D_fake)
        out.append(list(np.vstack((loss_D_real, loss_D_fake)).mean(axis=0)))
        out.append(loss_G)
    if return_accuracy:
        out.append(accuracy_real)
        out.append(accuracy_fake)
        out.append(list(np.vstack((accuracy_real, accuracy_fake)).mean(axis=0)))
    if return_score:
        out.append(score_real)
        out.append(score_fake)
        out.append(list(np.vstack((score_real, score_fake)).mean(axis=0)))
    if return_lr:
        out.append(lr_used_D)
        out.append(lr_used_G)
    out = tuple(out)
    if len(out) == 1:
        out = out[0]
    elif len(out) == 0:
        out = None
    return out
# ------------------------------------------------------------------------------

# ===== WGAN - GP =====
# ------------------------------------------------------------------------------
def gradient_penalty(model_D, X_real, X_fake, weight):
    """
    Computes gradient penalty (for WGAN).

    Parameters
    ----------
    model_D : network
        model (network) for the discriminator
    
    X_real, X_fake: tensor
        batch of real and fake data (of same size) respectively;
        size along axis 0 is the number of examples in the batch
    
    weight: float
        weight of the gradient penalty

    Returns
    -------
    gp : float (tensor)
        gradient penalty (mean of over all batch members), 
        weighted by the factor `weight`
    """
    device = X_real.device
    batch_size = X_real.shape[0]

    # Draw batch_size number in [0,1] (uniform) 
    # and store them in a tensor of shape (batch_size, 1, ..., 1) (same number of dimension as X_real, X_fake)
    t = torch.rand(batch_size, *((X_real.ndim-1)*[1])).to(device)

    # Expand (by copying numbers) in a tensor that fits shape of X_real, X_fake
    t = t.expand_as(X_real)
        
    # Interpolate between real data and fake data
    X_hat = t * X_real + (1-t)*X_fake
        
    # Get result of discriminator for interpolated examples
    X_hat_out = model_D(X_hat)

    # Compute gradients d(X_hat_out)/d(X_hat)
    #    X_hat_out of shape (batch_size, 1)
    #    gradients of shape X_hat.shape
    #    gradients[i, ...] = d(X_hat_out[i])/d(X_hat[i]) of shape X_hat[i].shape
    #       gradient of the i-th member
    gradients = torch.autograd.grad(
            outputs=X_hat_out,
            inputs=X_hat,
            grad_outputs=torch.ones_like(X_hat_out).to(device),
            create_graph=True, 
            retain_graph=True,
        )[0]
    # create and retain the graph for updating model_D further

    # Compute norm of gradient of each member
    gradients_norm = gradients.view(-1, 1).norm(p=2, dim=1) # p-norm along dim 1
    # torch.sqrt((gradients.view(-1, 1)**2).sum(dim=1)) # equiv.

    # Return gradient penalty (mean over each member), weighted
    return weight * ((gradients_norm - 1) ** 2).mean()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def train_wgan(
        model_D, 
        model_G,
        optimizer_D, 
        optimizer_G, 
        data_loader,
        gp_weight,
        update_D_n_steps=1,
        lr_scheduler_D=None,
        lr_scheduler_G=None,
        return_lr=False,
        return_loss=True,
        num_epochs=10,
        print_epoch=1,
        save_fake_epoch=0,
        z_fixed=None,
        save_fake_file_fmt='./fake_{:04d}.pt',
        device=torch.device('cpu')):
    """
    Trains a WGAN-GP model.

    Parameters
    ----------
    model_D : network
        model (network) for the discriminator
    
    model_G : network
        model (network) for the generator
    
    optimizer_D, optimizer_G : optimizer
        updater, torch optimizer for network model_D, model_G (resp.), e.g. 
        `torch.optim.SGD(net.parameters(), weight_decay=0.001, lr=0.03)`
    
    data_loader : data loader
        yielding mini-batches (x, n_nodes) from a data set, with:

            - B: current batch size
            - N: data_set.max_n_nodes
            - P: data_set.max_prev_node

            - x: tensor of shape (B, N, P+1)
                x[k]: k-th input of the mini-batch (encoded adjacency matrix)
            - n_nodes: tensor of shape (B, )
                n_nodes[k]: number of nodes taken into account in the encoding
                `x[k]`

    gp_weight : float
        gradient penalty weight
    
    update_D_n_steps : int, default: 1
        number of updates of the discriminator before updating of the generator,
        i.e. the generator is updated every `update_D_n_steps` mini-batches
    
    lr_scheduler_D, lr_scheduler_G : scheduler, optional
        learning rate scheduler for network model_D, model_G (resp.), e.g. 
        `torch.optim.lr_scheduler.*`
    
    return_lr : bool, default: `False`
        if `True`: return two sequences of learning rates used:
        - dicriminator lr
        - generator lr
    
    return_loss : bool, default: `True`
        if `True`: return losses (see below)
    
    num_epochs : int, default: 10
        number of epochs
    
    print_epoch : int, default: 10
        result of every `print_epoch` epoch is displayed in stdout, if 
        `print_epoch > 0`
    
    save_fake_epoch : int, default: 0
        if `save_fake_epoch > 0`, at each generated fake data, using z_fixed as input and 
        a tensor of generated fake data, `model_G(z_fixed)`, at every `save_fake_epoch` epoch 
        is saved (written on the disk), provided `z_fixed` is specified (not `None`)
    
    z_fixed : tensor of 2 dimensions, optional
        tensor of random noise in N(0,1), used to generate fake data to save (see 
        `save_fake_epoch`), its shape should be (m, latent_dim) where m is the number of 
        fake data to be generated and latent_dim the dimension of the latent space; 
        unused if `save_fake_epoch<=0`
    
    save_fake_file_fmt : str, default: './fake_{:04d}.pt'
        string for filename (including path) for saving generated fake data 
        (see `save_fake_epoch`), at epoch i the file will be `save_fake_file_fmt.format(i)`
    
    device : torch device, default: torch.device('cpu')
        device on which the model is trained
    
    Returns
    -------
    loss_D_real : list, optional
        returned if `return_loss=True`, discriminator loss over real data 
        of every epoch, list of floats of length `num_epochs`
    
    loss_D_fake : list, optional
        returned if `return_loss=True`, discriminator loss over fake data 
        of every epoch, list of floats of length `num_epochs`
    
    loss_D_gp : list, optional
        returned if `return_loss=True`, gradient penalty added
        to "loss_D_fake - loss_D_real", which is an approximation of 
        minus Wasserstein-1 distance between real and fake distribution,
        when optimizing the discriminator (maximizing Wasserstein distance, 
        and ensuring Lipschitz-1 condition with gradient penalty),
        list of floats of length `num_epochs`
    
    lr_used_D : list, optional
        returned if `return_lr=True`, learning rate for the discriminator
        used at each epoch, list of floats of length `num_epochs`,        
    
    lr_used_G : list, optional
        returned if `return_lr=True`, learning rate for the generator
        used at each epoch, list of floats of length `num_epochs`,        
    
    Notes
    -----
    With n the total number of members in the training set
    the parameters batch_size and update_D_n_steps should satified:
    
    - n = m * batch_size (i.e. n multiple of batch_size)
    - m = k * update_D_n_steps (i.e. m multiple of update_D_n_steps)
    """
    fname = 'train_wgan'

    # Copy networks to device
    model_D.to(device)
    model_G.to(device)

    model_D.train() # set the Discriminator network in training mode
    model_G.train() # set the Generator     network in training mode

    # Dimension of latent space
    latent_dim = model_G.latent_dim

    # Initialize list for loss
    if return_loss:
        loss_D_real, loss_D_fake, loss_D_gp = [], [], []
        loss_D_real_epoch, loss_D_fake_epoch, loss_D_gp_epoch = 0.0, 0.0, 0.0 # reset loss of one epoch

    # Initialize list for lr
    if return_lr:
        lr_used_D = []
        lr_used_G = []

    # Check save_fake_epoch and z_fixed
    if save_fake_epoch:
        if z_fixed is None or z_fixed.ndim != 2 or z_fixed.shape[1] != latent_dim:
            print(f'ERROR ({fname}): `z_fixed` not valid')
            return
        z_fixed = z_fixed.to(device)

    # Train the network
    num_steps = 0 # number of mini-batch steps (over all epochs)
    for epoch in range(num_epochs):
        # Train GAN through every mini-batch
        train_D_len = 0  # reset train length for discriminator
        for d in data_loader:
            X = d[0]
            # n_nodes = d[1]
            
            num_steps += 1
            # if num_steps % update_D_n_steps == 0:
            #     print(f' ... Step: {num_steps}: train discriminator and generator')
            # else:
            #     print(f' ... Step: {num_steps}: train discriminator')

            # one mini-batch
            X = X.to(device) # copy to device

            # Train the Discriminator
            # -----------------------
            X_real_out = model_D(X) # Discriminator forward for real data            

            # Generate fake data via the Generator
            z = torch.randn(len(X), latent_dim, device=device)
            X_fake = model_G(z) # fake data via Generator forward
            X_fake = torch.sigmoid(X_fake) # OR: X_fake = torch.nn.Sigmoid()(X_fake)
            X_fake_out = model_D(X_fake.detach()) # Discriminator forward for fake data
                                                  # - Use detach() to not track the gradient
                                                  # because the Discriminator is trained!

            # Compute Discriminator loss for the current mini-batch (mb)
            loss_D_real_mb = X_real_out.mean() # loss for real data
            loss_D_fake_mb = X_fake_out.mean() # loss for fake data 
            loss_D_gp_mb = gradient_penalty(model_D, X, X_fake, gp_weight)
            loss_D_mb = - loss_D_real_mb + loss_D_fake_mb + loss_D_gp_mb
            
            # Update Discriminator parameters
            optimizer_D.zero_grad() # reset gradient in optimized tensors of Discriminator
            loss_D_mb.backward()    # compute gradient of Discriminator loss wrt to Discriminator parameters
            optimizer_D.step()      # one step in Discriminator training (update)

            if return_loss:
                with torch.no_grad():
                    loss_D_real_epoch += loss_D_real_mb.item()*X.shape[0]
                    loss_D_fake_epoch += loss_D_fake_mb.item()*X.shape[0]
                    loss_D_gp_epoch += loss_D_gp_mb.item()*X.shape[0]

            train_D_len += X.shape[0]

            if num_steps % update_D_n_steps == 0:
                # Train the Generator
                # -------------------
                # Generate fake data via the Generator
                z = torch.randn(len(X), latent_dim, device=device) 
                X_fake = model_G(z) # fake data via Generator forward
                X_fake = torch.sigmoid(X_fake) # OR: X_fake = torch.nn.Sigmoid()(X_fake)
                X_fake_out = model_D(X_fake) # Discriminator forward for fake data

                # Compute Generator loss for the current mini-batch (mb)
                loss_G_mb = -X_fake_out.mean()
                
                # Update Generator parameters
                optimizer_G.zero_grad() # reset gradient in optimized tensors of Generator
                loss_G_mb.backward()    # compute gradient of Generator loss wrt to Generator parameters
                optimizer_G.step()      # one step in Generator training (update)

        if return_lr:
            lr_used_D.append(optimizer_D.param_groups[0]['lr'])
            lr_used_G.append(optimizer_G.param_groups[0]['lr'])
        if lr_scheduler_D:
            # lr_used_D.append(lr_scheduler_D.get_last_lr()[0])
            lr_scheduler_D.step()
        if lr_scheduler_G:
            # lr_used_G.append(lr_scheduler_G.get_last_lr()[0])
            lr_scheduler_G.step()

        if return_loss:
            loss_D_real.append(loss_D_real_epoch/train_D_len)
            loss_D_fake.append(loss_D_fake_epoch/train_D_len)
            loss_D_gp.append(loss_D_gp_epoch/train_D_len)
            loss_D_real_epoch, loss_D_fake_epoch, loss_D_gp_epoch = 0.0, 0.0, 0.0 # reset loss of one epoch

        if print_epoch > 0 and epoch % print_epoch == 0:
            # Print result of current epoch
            s = f'epoch {epoch+1} of {num_epochs}'
            if return_loss:
                s = s + f', loss_D (-real, fake, gp): {-loss_D_real[-1]:9.4f}, {loss_D_fake[-1]:9.4f}, {loss_D_gp[-1]:9.4f}, loss_D fake - real: {loss_D_fake[-1] - loss_D_real[-1]:9.4f}'
            print(s)

        if save_fake_epoch > 0 and epoch%save_fake_epoch == 0:
            with torch.no_grad():
                X_fake_save = model_G(z_fixed)
                X_fake_save = torch.sigmoid(X_fake_save).to('cpu')
                torch.save(X_fake_save, save_fake_file_fmt.format(epoch))

    # Set networks on cpu
    model_D.to(torch.device('cpu'))
    model_G.to(torch.device('cpu'))

    out = []
    if return_loss:
        out.append(loss_D_real)
        out.append(loss_D_fake)
        out.append(loss_D_gp)
    if return_lr:
        out.append(lr_used_D)
        out.append(lr_used_G)
    out = tuple(out)
    if len(out) == 1:
        out = out[0]
    elif len(out) == 0:
        out = None
    return out
# ------------------------------------------------------------------------------

# ===== GENERATE GRAPH USING GENERATOR OF THE GAN =====
# ------------------------------------------------------------------------------
def generate_graph_gan(
        model_G, 
        n_graph=1,
        sample_encoded=True,
        treshold_encoded=0.5,
        force_node0=False,
        return_encoded=False,
        device=torch.device('cpu')):
    """
    Generates one or several graph(s) using the generator of a GAN.
    
    The generator provides "raw" encoded adjacency matrices, filled
    with floats in (0, 1) interpreted as probabilities (on each row:
    probability to have the next node, followed by probabilities to 
    have an edge linking that nodes to the previous ones).

    Then, encoded adjacency matrices (binary with value 0 and 1) are
    obtained from the "raw" ones, either by sampling or by 
    thresholding.

    Parameters
    ----------
    model_G : network
        model (network) for the generator (of the gan)
    
    n_graph : int, default: 1
        number of graph to be generated
    
    sample_encoded : bool, default: `True`
        the encoded adjacency matrices are obtained from the 
        "raw" ones by sampling (True) or by thresholding (False)
    
    threshold_encoded : float, default: 0.5
        used if `sample_encoded=False`: threshold value to get
        the encoded adjacency matrices (binary with value 0 and 1) 
        from the "raw" ones: values greater than or equal to 
        `threshold_encoded` become ones
    
    force_node0 : bool, default: `False`
        if `True`: force creation of node 0
    
    return_encoded : bool, default: `False`
        if `True`: the encoded adjacency matrix is returned
    
    device : torch device, default: torch.device('cpu')
        device on which the network is trained
    
    Returns
    -------
    G_list : list of networkx.Graph object of length `n_graph`
        generated graphs:

        - G_list[k]: k-th graph
    
    X : 3d tensor of floats, optional
        of shape `(n_graph, model_G.max_n_nodes, model_G.max_prev_node+1)`,
        "raw" encoded adjacency matrices obtained via the generator (`model_G`),
        float values in (0, 1), interpreted as probabilities (see above)
    """
    fname = 'generate_graph_gan'

    # Copy models to device
    model_G.to(device)

    # Set models in evaluation mode
    model_G.eval()

    # Generate encoded graph via the Generator
    with torch.no_grad(): 
        z = torch.randn(n_graph, model_G.latent_dim, device=device) 
        X = model_G(z)
        X = torch.sigmoid(X) # OR: X = torch.nn.Sigmoid()(X)

    # Set model on cpu
    model_G.to(torch.device('cpu'))

    # Get encoded adjacency matrices (binary)
    if sample_encoded:
        adj_prev_array_all = (torch.rand_like(X) < X).to(torch.int).to(torch.device('cpu')).numpy()
    else:
        adj_prev_array_all = torch.ge(X, treshold_encoded).to(torch.int).to(torch.device('cpu')).numpy()

    if force_node0:
        adj_prev_array_all[:, 0, 0] = 1

    # Get graphs
    G_list = [decode_graph(adj_prev_array) for adj_prev_array in adj_prev_array_all]

    if return_encoded:
        return G_list, X.to(torch.device('cpu'))
    else: 
        return G_list
# ------------------------------------------------------------------------------


###### OLD #####
# =============================================================================
# GAN model for graph generation accounting for node features
# =============================================================================
# ------------------------------------------------------------------------------



################################################################################
# ------------------------------------------------------------------------------
# # Train the generator, then the discriminator (using same fake data).
# def train_gan(
#         model_D, 
#         model_G,
#         optimizer_D, 
#         optimizer_G, 
#         data_loader,
#         lr_scheduler_D=None,
#         lr_scheduler_G=None,
#         return_lr=False,
#         return_loss=True,
#         return_accuracy=False,
#         return_score=False,
#         num_epochs=10,
#         print_epoch=1,
#         save_fake_epoch=0,
#         z_fixed=None,
#         save_fake_file_fmt='./fake_{:04d}.pt',
#         device=torch.device('cpu')):
#     """
#     Trains a GAN model.
    
#     Train the generator, then the discriminator (using same fake data).

#     Parameters
#     ----------
#     model_D : network
#         model (network) for the discriminator
#     model_G : network
#         model (network) for the generator
#     optimizer_D, optimizer_G : optimizer
#         updater, torch optimizer for network model_D, model_G (resp.), e.g. 
#         `torch.optim.SGD(net.parameters(), weight_decay=0.001, lr=0.03)`
#     data_loader : data loader
#         yielding mini-batches (x, n_nodes) from a data set, with:
#             - B: current batch size
#             - N: data_set.max_n_nodes
#             - P: data_set.max_prev_node
#         - x: tensor of shape (B, N, P+1):
#             x[k]: k-th input of the mini-batch (encoded adjacency matrix)
#         - n_nodes: tensor of shape (B, )
#             n_nodes[k]: number of nodes taken into account in the encoding
#             `x[k]`
#     lr_scheduler_D, lr_scheduler_G : scheduler, optional
#         learning rate scheduler for network model_D, model_G (resp.), e.g. 
#         `torch.optim.lr_scheduler.*`
#     return_lr : bool, default: `False`
#         if `True`: return two sequences of learning rates used:
#         - dicriminator lr
#         - generator lr
#     return_loss : bool, default: `True`
#         if `True`: return four sequences of losses:
#         - discriminator loss for real data
#         - discriminator loss for fake data
#         - discriminator loss (mean)
#         - generator loss
#     return_loss : bool, default: `True`
#         if `True`: return losses (see below)
#     return_accuracy : bool, default: `False`
#         if `True`: return three sequences of accuracies:
#         - accuracy of the discriminator for real data
#         - accuracy of the discriminator for fake data
#         - accuracy of the discriminator (mean of accuracy for real and 
#         fake data);
#         accuracy is defined as the rate of true prediction (real or fake)
#     return_score : bool, default: `False`
#         if `True`: return three sequences of scores:
#         - score of the discriminator for real data
#         - score of the discriminator for fake data
#         - score of the discriminator (mean of score for real and 
#         fake data);
#         score is defined as the mean score over single input; if p is the
#         predicted probability to be real for a single input, the corresponding
#         score is s=p for real input and s=(1-p) for fake input
#     num_epochs : int, default: 10
#         number of epochs
#     print_epoch : int, default: 10
#         result of every `print_epoch` epoch is displayed in stdout, if 
#         `print_epoch > 0`
#     save_fake_epoch : int, default: 0
#         if `save_fake_epoch > 0`, at each generated fake data, using z_fixed as input and 
#         a tensor of generated fake data, `model_G(z_fixed)`, at every `save_fake_epoch` epoch 
#         is saved (written on the disk), provided `z_fixed` is specified (not `None`)
#     z_fixed : tensor of 2 dimensions, optional
#         tensor of random noise in N(0,1), used to generate fake data to save (see 
#         `save_fake_epoch`), its shape should be (m, latent_dim) where m is the number of 
#         fake data to be generated and latent_dim the dimension of the latent space; 
#         unused if `save_fake_epoch<=0`
#     save_fake_file_fmt : str, default: './fake_{:04d}.pt'
#         string for filename (including path) for saving generated fake data 
#         (see `save_fake_epoch`), at epoch i the file will be `save_fake_file_fmt.format(i)`
#     device : torch device, default: torch.device('cpu')
#         device on which the model is trained
    
#     Returns
#     -------
#     loss_D_real : list, optional
#         returned if `return_loss=True`, discriminator loss over real data 
#         of every epoch, list of floats of length `num_epochs`
#     loss_D_fake : list, optional
#         returned if `return_loss=True`, discriminator loss over fake data 
#         of every epoch, list of floats of length `num_epochs`
#     loss_D : list, optional
#         returned if `return_loss=True`, discriminator loss of every epoch, 
#         list of floats of length `num_epochs`
#     loss_G : list, optional
#         returned if `return_loss=True`, generator loss of every epoch, 
#         list of floats of length `num_epochs`
#     accuracy_real : list, optional
#         returned if `return_accuracy=True`, accuracy of the discriminator
#         for real data (number of true prediction over total number of 
#         predictions) of every epoch, 
#         list of floats of length `num_epochs`
#     accuracy_fake : list, optional
#         returned if `return_accuracy=True`, accuracy of the discriminator
#         for fake, i.e. generated, data (number of true prediction over total 
#         number of predictions) of every epoch, 
#         list of floats of length `num_epochs`
#     accuracy : list, optional
#         returned if `return_accuracy=True`, accuracy of the discriminator,
#         mean of `accuracy_real` and `acccuracy_fake`;
#         list of floats of length `num_epochs`
#     score_real : list, optional
#         returned if `return_score=True`, score of the discriminator
#         for real data (mean of predicted probabilities);
#         list of floats of length `num_epochs`
#     score_fake : list, optional
#         returned if `return_score=True`, score of the discriminator
#         for fake data (one minus mean of predicted probabilities);
#         list of floats of length `num_epochs`
#     score : list, optional
#         returned if `return_score=True`, score of the discriminator,
#         mean of `score_real` and `score_fake`;
#         list of floats of length `num_epochs`
#     lr_used_D : list, optional
#         returned if `return_lr=True`, learning rate for the discriminator
#         used at each epoch, list of floats of length `num_epochs`,        
#     lr_used_G : list, optional
#         returned if `return_lr=True`, learning rate for the generator
#         used at each epoch, list of floats of length `num_epochs`,        
    
#     Notes
#     -----
#     With n the total number of members in the training set
#     the parameters batch_size and update_D_n_steps should satified:
#         - n = m * batch_size (i.e. n multiple of batch_size)
#         - m = k * update_D_n_steps (i.e. m multiple of update_D_n_steps)
#     """
#     fname = 'train_gan'

#     # Copy networks to device
#     model_D.to(device)
#     model_G.to(device)

#     model_D.train() # set the Discriminator network in training mode
#     model_G.train() # set the Generator     network in training mode

#     # Dimension of latent space
#     latent_dim = model_G.latent_dim

#     # Initialize list for loss
#     if return_loss:
#         loss_D_real, loss_D_fake, loss_G = [], [], []
#         loss_D_real_epoch, loss_D_fake_epoch, loss_G_epoch = 0.0, 0.0, 0.0 # reset loss of one epoch

#     # Initialize list for accuracy
#     if return_accuracy:
#         accuracy_real, accuracy_fake = [], []
#         accuracy_real_epoch, accuracy_fake_epoch = 0.0, 0.0 # reset accuracy of one epoch

#     # Initialize list for score
#     if return_score:
#         score_real, score_fake = [], []
#         score_real_epoch, score_fake_epoch = 0.0, 0.0 # reset score of one epoch

#     # Initialize list for lr
#     if return_lr:
#         lr_used_D = []
#         lr_used_G = []

#     # Check save_fake_epoch and z_fixed
#     if save_fake_epoch:
#         if z_fixed is None or z_fixed.ndim != 2 or z_fixed.shape[1] != latent_dim:
#             print(f'ERROR ({fname}): `z_fixed` not valid')
#             return
#         z_fixed = z_fixed.to(device)

#     # Train the network
#     for epoch in range(num_epochs):
#         # Train GAN through every mini-batch
#         train_len = 0  # reset train length
#         for d in data_loader:
#             X = d[0]
#             # n_nodes = d[1]
            
#             # one mini-batch
#             X = X.to(device) # copy to device

#             # Create tensor of labels (real and fake)
#             real_label = torch.full((len(X), 1), 1, dtype=X.dtype, device=device)
#             fake_label = torch.full((len(X), 1), 0, dtype=X.dtype, device=device)
            
#             # Generate fake data via the Generator
#             z = torch.randn(len(X), latent_dim, device=device) 
#             X_fake = model_G(z) # fake data via Generator forward
#             X_fake = torch.sigmoid(X_fake) # OR: X_fake = torch.nn.Sigmoid()(X_fake)

#             # Train the Generator
#             # -------------------
#             X_fake_out = model_D(X_fake) # Discriminator forward for fake data
#             X_fake_out = torch.sigmoid(X_fake_out)

#             # Compute Generator loss for the current mini-batch (mb) by giving fake data and real label
#             loss_G_mb = torch.nn.functional.binary_cross_entropy(X_fake_out, real_label)
#             # loss_G_mb = torch.nn.BCELoss()(X_fake_out, real_label)
            
#             # Update Generator parameters
#             optimizer_G.zero_grad() # reset gradient in optimized tensors of Generator
#             loss_G_mb.backward()    # compute gradient of Generator loss wrt to Generator parameters
#             optimizer_G.step()      # one step in Generator training (update)

#             # Train the Discriminator
#             # -----------------------
#             X_real_out = model_D(X) # Discriminator forward for real data            
#             X_real_out = torch.sigmoid(X_real_out) # OR: X_real_out = torch.nn.Sigmoid()(X_real_out)

#             # Compute Discriminator loss for the current mini-batch (mb)
#             # - for fake data, use detach() to not track the gradient because the Discriminator is trained!
#             X_fake_out = model_D(X_fake.detach()) # Discriminator forward for fake data
#             X_fake_out = torch.sigmoid(X_fake_out)
#             loss_D_real_mb = torch.nn.functional.binary_cross_entropy(X_real_out, real_label) # loss for real data
#             loss_D_fake_mb = torch.nn.functional.binary_cross_entropy(X_fake_out, fake_label) # loss for fake data 
#             # loss_D_real_mb = torch.nn.BCELoss()(X_real_out, real_label) # loss for real data
#             # loss_D_fake_mb = torch.nn.BCELoss()(X_fake_out, fake_label) # loss for fake data 
#             loss_D_mb = 0.5*(loss_D_real_mb + loss_D_fake_mb)

#             # Update Discriminator parameters
#             optimizer_D.zero_grad() # reset gradient in optimized tensors of Discriminator
#             loss_D_mb.backward()    # compute gradient of Discriminator loss wrt to Discriminator parameters
#             optimizer_D.step()      # one step in Discriminator training (update)

#             if return_loss:
#                 with torch.no_grad():
#                     loss_D_real_epoch += loss_D_real_mb.item()*X.shape[0]
#                     loss_D_fake_epoch += loss_D_fake_mb.item()*X.shape[0]
#                     loss_G_epoch += loss_G_mb.item()*X.shape[0]

#             if return_accuracy:
#                 with torch.no_grad():
#                     accuracy_real_epoch += (X_real_out.view(-1) >  0.5).sum().item()
#                     accuracy_fake_epoch += (X_fake_out.view(-1) <= 0.5).sum().item()

#             if return_score:
#                 with torch.no_grad():
#                     score_real_epoch += X_real_out.sum().item()
#                     score_fake_epoch += X_fake_out.sum().item()

#             train_len += X.shape[0]

#         if return_lr:
#             lr_used_D.append(optimizer_D.param_groups[0]['lr'])
#             lr_used_G.append(optimizer_G.param_groups[0]['lr'])
#         if lr_scheduler_D:
#             # lr_used_D.append(lr_scheduler_D.get_last_lr()[0])
#             lr_scheduler_D.step()
#         if lr_scheduler_G:
#             # lr_used_G.append(lr_scheduler_G.get_last_lr()[0])
#             lr_scheduler_G.step()

#         if return_loss:
#             loss_D_real.append(loss_D_real_epoch/train_len)
#             loss_D_fake.append(loss_D_fake_epoch/train_len)
#             loss_G.append(loss_G_epoch/train_len)
#             loss_D_real_epoch, loss_D_fake_epoch, loss_G_epoch = 0.0, 0.0, 0.0 # reset loss of one epoch

#         if return_accuracy:
#             accuracy_real.append(accuracy_real_epoch/train_len)
#             accuracy_fake.append(accuracy_fake_epoch/train_len)
#             accuracy_real_epoch, accuracy_fake_epoch = 0.0, 0.0 # reset accuracy of one epoch

#         if return_score:
#             score_real.append(score_real_epoch/train_len)
#             score_fake.append(1.0 - score_fake_epoch/train_len)
#             score_real_epoch, score_fake_epoch = 0.0, 0.0 # reset accuracy of one epoch

#         if print_epoch > 0 and epoch % print_epoch == 0:
#             # Print result of current epoch
#             s = f'epoch {epoch+1} of {num_epochs}'
#             if return_loss:
#                 s = s + f', loss_G: {loss_G[-1]:.4f}, loss_D (real, fake, mean): {loss_D_real[-1]:.4f}, {loss_D_fake[-1]:.4f}, {0.5*(loss_D_real[-1]+loss_D_fake[-1]):.4f}'
#             if return_accuracy:
#                 s = s + f', accuracy: {accuracy_real[-1]:.4f}, {accuracy_fake[-1]:.4f}, {0.5*(accuracy_real[-1]+accuracy_fake[-1]):.4f}'
#             if return_score:
#                 s = s + f', score: {score_real[-1]:.4f}, {score_fake[-1]:.4f}, {0.5*(score_real[-1]+score_fake[-1]):.4f}'
#             print(s)

#         if save_fake_epoch > 0 and epoch%save_fake_epoch == 0:
#             with torch.no_grad():
#                 X_fake_save = model_G(z_fixed)
#                 X_fake_save = torch.sigmoid(X_fake_save).to('cpu')
#                 torch.save(X_fake_save, save_fake_file_fmt.format(epoch))

#     # Set networks on cpu
#     model_D.to(torch.device('cpu'))
#     model_G.to(torch.device('cpu'))

#     out = []
#     if return_loss:
#         out.append(loss_D_real)
#         out.append(loss_D_fake)
#         out.append(list(np.vstack((loss_D_real, loss_D_fake)).mean(axis=0)))
#         out.append(loss_G)
#     if return_accuracy:
#         out.append(accuracy_real)
#         out.append(accuracy_fake)
#         out.append(list(np.vstack((accuracy_real, accuracy_fake)).mean(axis=0)))
#     if return_score:
#         out.append(score_real)
#         out.append(score_fake)
#         out.append(list(np.vstack((score_real, score_fake)).mean(axis=0)))
#     if return_lr:
#         out.append(lr_used_D)
#         out.append(lr_used_G)
#     out = tuple(out)
#     if len(out) == 1:
#         out = out[0]
#     elif len(out) == 0:
#         out = None
#     return out
# # ------------------------------------------------------------------------------
