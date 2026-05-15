#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for Graph RNN.
"""

# -------------------------------------------------------------------------
#   Author: Julien Straubhaar
#   Year: 2024
#   Company: University of Neuchâtel
#
#   Copyright (c) 2024 Julien Straubhaar
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# -------------------------------------------------------------------------

import numpy as np
import scipy
import networkx

import torch

#  import pyvista as pv
# import matplotlib.pyplot as plt

# # import from package 'geone'
# import geone as gn

# NOTE: load graph_utils.py

# Standalone fallback — overwritten when general_utils.py is exec()'d first
def default_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# =============================================================================
# Base RNN model
# =============================================================================
# -----------------------------------------------------------------------------
# RNN model
# ---------
class RNN_model(torch.nn.Module):
    """ 
    Class defining a RNN model.
    
    Parameters
    ----------
    input_size : int
        size of input at each time step (number of input features)
    
    hidden_size : int
        size of hidden state variable(s) in the RNN cells       
    
    num_layers : int
        number of layers of RNN cells (stacked)
    
    rnn_type : str {'RNN', 'GRU', 'LSTM'}; default: 'GRU'
        type of RNN cell
    
    dropout : float in [0, 1), default: 0.0
        probability for dropout for output of all rnn layers except the last one 
        (`dropout` set to 0.0 if `num_layers=1`)
    
    embed_input : bool, default: `False`
        - if `True`: input is embedded (using a fully connected layer + ReLU), before \
        using it to feed first layer of rnn cells
        - if `False`: input is use as given to feed first layer of rnn cells
    
    embed_input_size : int, optional
        size of embedded input, must be specified if `embed_input=True`
    
    has_output : bool, default: `False`
        - if `True`: output is computed from the output of the rnn cells at last layer
        - if `False`: output is set to the output of the rnn cells at last layer
    
    output_size : int, optional
        size of output, must be specified if `has_output=True`;
        number of output features (set automatically if needed)
    
    embed_output : bool, default: `False`
        used if `has_ouput=True`:

        - if `True`: output of the rnn cells at last layer is embedded \
        (using a fully connected layer + ReLU) before computing final output
        - if `False`: output of the rnn cells at last layer is is used \
        to compute final output

    Notes
    -----
    With:

    - T: 
        number of time steps
    - L: 
        number of layers (`num_layers`)
    - x[i]:     
        input for time step i, for i = 0, ..., T-1
        x[i] of size `input_size`
    - x_emb[i]: 
        embedded input for time step i, for i = 0, ..., T-1
        x_emb[i] of size `embed_input_size`;
        optional: not used if `embed_input = False`
    - h[i, j]:
        hidden variable for time step i, at layer j,
        for i = 0, ..., T-1, for j = 0, ..., L-1; and
        h[T, j]: hidden variable after time step T-1;
        h[i, j] of size `hidden_size`
    - y_emb[i]: 
        embedded output for time step i, for i = 0, ..., T-1
        y_emb[i] of size `embed_output_size`;
        optional: not used if `embed_output = False`
    - y[i]: 
        output for time step i, for i = 0, ..., T-1
        y[i] of size `output_size`;
        optional: not used if `has_output = False`, in such 
        case the hidden variable at the exit of the last layer is 
        returned as ouput
    
    the architecture of the model is::

            [optional       y[0]                                                         y[T-1]          ]     
            [  [optional     ^                                                             ^         ]   ]
            [  [             |                                                             |         ]   ]
            [  [             | Fully connected layer                                       |         ]   ]
            [  [             |                                                             |         ]   ]
            [  [          y_emb[0]                                                      y_emb[T-1]   ]   ]
            [                ^                                                             ^             ]
            [                |                                                             |             ]
            [                | Fully connected layer + ReLU                                |             ]
            [                |                                                             |             ]
                            h[1, L-1]                       ...                         h[T, L-1]
                              ^                             ^                              ^     
                              |                             |                              |     
                         +----------+                  +----------+                   +----------+          
                         | RNN cell |                  |          |                   | RNN cell |          
        -- h[0, L-1] --> |          |--> h[1, L-1] --> |          | --> h[T-1, 0] --> |          | --> h[T, L-1]
                         |          |                  |          |                   |          |
                         +----------+                  +----------+                   +----------+
                              ^                             ^                              ^   
                              |                             |                              |   
                           h[1, L-2]                       ...                          h[T, L-2]
                              ^                             ^                              ^   
                              |                             |                              |
                         +----------+                  +----------+                   +----------+
                         |          |                  |          |                   |          |
                         |   ...    | --> ... -->      |    ...   | --> ... -->       |   ...    | --> ...
                         |          |                  |          |                   |          |
                         +----------+                  +----------+                   +----------+
                             ^                              ^                              ^   
                             |                              |                              |   
                          h[1, 0]                          ...                          h[T, 0]
                             ^                              ^                              ^   
                             |                              |                              |
                         +----------+                  +----------+                   +----------+
                         | RNN cell |                  |          |                   | RNN cell |
        -- h[0, 0]   --> |          | --> h[1, 0] -->  |   ...    | --> h[T-1, 0] --> |          | --> h[T, 0]
                         |          |                  |          |                   |          |
                         +----------+                  +----------+                   +----------+
                             ^                                                             ^       
                             |                                                             |       
                [optional  x_emb[0]                                                      x_emb[T-1]  ]
                [            ^                                                             ^         ]
                [            |                                                             |         ]
                [            | Fully connected layer + ReLU                                |         ]
                [            |                                                             |         ]
                            x[0]                                                          x[T-1]   
    
    - The class defines the model for one time step (i.e. T is not a parameter of the model). \
    The model consists of `num_layers`(number L above) stacked RNN cell(s), where the \
    the parameter `num_layers` is the number L above.
    - The `forward` returns y = y[0], ... y[T-1], and h = h[T, 0], ...h[T, L-1]
    """
    def __init__(self, 
                 input_size,
                 hidden_size, 
                 num_layers=1,
                 rnn_type='GRU',
                 dropout=0.0,
                 embed_input=False,
                 embed_input_size=None,
                 has_output=False,
                 output_size=None,
                 embed_output=False,
                 embed_output_size=None):
        """Constructor method.
        """

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if self.num_layers == 1:
            self.dropout = 0.0
        else:
            self.dropout = dropout

        self.embed_input = embed_input
        if self.embed_input:
            self.rnn_input_size = embed_input_size
            self.in_module = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, self.rnn_input_size),
                torch.nn.ReLU()
            )
            #
            # self.fc_embed_in = torch.nn.Linear(self.input_size, self.rnn_input_size)
            # self.relu_in = torch.nn.ReLU()
        else:
            self.rnn_input_size = self.input_size

        self.rnn_type = rnn_type
        if self.rnn_type == 'RNN':
            self.rnn = torch.nn.RNN(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers, bias=True, 
                                    dropout=self.dropout, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = torch.nn.GRU(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers, bias=True, 
                                    dropout=self.dropout, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = torch.nn.LSTM(self.rnn_input_size, self.hidden_size, num_layers=self.num_layers, bias=True, 
                                     dropout=self.dropout, batch_first=True)
            # with this type: a 2-tuple (h, c), "(hidden variable, cell state)" is used instead of h
        else:
            print('ERROR: init `RNN_model`: `rnn_type` not known')
            return

        self.has_output = has_output
        if self.has_output:
            self.output_size = output_size
            self.embed_output = embed_output
            if self.embed_output:
                self.embed_output_size = embed_output_size
                self.out_module = torch.nn.Sequential(    
                    torch.nn.Linear(self.hidden_size, self.embed_output_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.embed_output_size, self.output_size)
                )
                #
                # self.embed_output_size = embed_output_size
                # self.fc_embed_out = torch.nn.Linear(self.hidden_size, self.embed_output_size)
                # self.relu_out = torch.nn.ReLU()
                # self.fc_out = torch.nn.Linear(self.embed_output_size, self.output_size)
            else:
                self.out_module = torch.nn.Sequential(    
                    torch.nn.Linear(self.hidden_size, self.output_size)
                )
                # self.out_module = torch.nn.Linear(self.hidden_size, self.output_size)
                #
                # self.fc_out = torch.nn.Linear(self.hidden_size, self.output_size)
        else:
            self.output_size = hidden_size

        self.init_weights()

    def forward(self, input, state, pack=False, pack_lens=None):
        """
        Forward (using `batch_first=True`).

        Parameters
        ----------
        input : tensor
            inputs, of shape (B, T, input_size) (or (T, input_size) for unbatched input),
            where B is the batch size, T the number of time steps, and input_size the number of 
            input features
        
        state : tensor, or tuple of tensors
            state variable(s):

                - h      : hidden state
                - (h, c) : (hidden state, cell state (memory)), for lstm cell (:class:`torch.nn.LSTM`), \
                           h and c of same size

            h (and c) of shape (B, num_layers, hidden_size) (or (num_layers, hidden_size) 
            if unbatched input)
        
        pack : bool, default: `False`
            - if `True`: `input` tensor has to be packed \
            (using `torch.nn.utils.rnn.pack_padded_sequence`) according to lengths given by \
            `pack_lens` tensor
            - if `False`: `input` tensor is used as given
        
        pack_lens : tensor, optional
            used (and must be specified) if `pack=True`: tensor of shape (B, ) containing the
            length (number of time steps) of each element of the batch, i.e. 
            input[i,0:pack_lens[i],...] is the i-th element of the batch to taken into account

        Returns
        -------
        output : tensor
            tensor of shape (B, T, output_size) (or (T, output_size) if unbatched input),
            where T is the number of time steps and output_size is the number of output features

        state : tensor, or tuple of tensors 
            same shape as state in input parameters, state variable(s) at the end of the sequence 
            (after T time steps) 
        """
        if self.embed_input:
            input = self.in_module(input)

        if pack:
            # pack input
            input = torch.nn.utils.rnn.pack_padded_sequence(input, pack_lens, batch_first=True)
            # -> sort the batches by descending length (number of time steps):
            # input is of type `PackedSequence` with
            # input.data: tensor of shape (sum(pack_lens), input_size)
            #    input.data[0:input.batch_sizes[0]] : 
            #       item (time step) 0 of all batches of len >=1
            #    input.data[input.batch_sizes[0]:input.batch_sizes[0]+input.batch_sizes[1]] : 
            #       item (time step) 1 of all batches of len >=2
            #    ...
            #    input.sorted_indices, input.unsorted_indices: allow to deal with the ordering
        output, state = self.rnn(input, state)

        if pack:
            # unpack output
            output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

        if self.has_output:
            output = self.out_module(output)

        return output, state

    def init_state(self, batch_size=0, device=None): # device='cpu' if None...
        """Initializes variable `state` (hidden state [, cell state]) with zeros."""
        if batch_size > 0:
            out_shape = self.num_layers, batch_size, self.hidden_size # batch_first=True is NOT applied on hidden and cell states!
        else:
            out_shape = self.num_layers, self.hidden_size

        if isinstance(self.rnn, torch.nn.LSTM):
            return torch.zeros(out_shape, device=device), torch.zeros(out_shape, device=device)
        else:
            return torch.zeros(out_shape, device=device)

    def init_weights(self, seed=None):
        """Initializes weights of the network."""
        if seed is not None:
            torch.random.manual_seed(seed)

        for name, param in self.rnn.named_parameters():
            # print('rnn...', name)
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param, gain=1.0)
                                                    # gain=nn.init.calculate_gain('sigmoid')
                                                    # gain=nn.init.calculate_gain('relu')
                                                    # ...

        if self.embed_input:
            for name, param in self.in_module.named_parameters():
                # print('in_module ...', name)
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    torch.nn.init.xavier_uniform_(param, gain=1.0)

        if self.has_output:
            for name, param in self.out_module.named_parameters():
                # print('out_module ...', name)
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    torch.nn.init.xavier_uniform_(param, gain=1.0)
# -----------------------------------------------------------------------------      

# =============================================================================
# Encoding / decoding adjacency matrix (not accounting for node features)
# =============================================================================
# -----------------------------------------------------------------------------
def encode_adj(adj_mat_csr, max_n_nodes=None, max_prev_node=None):
    """
    Encodes an adjacency matrix in csr format of a graph.

    Let n_nodes be the number of nodes in the graph. The "encoded adjacency matrix" 
    consists of a 2d arrray `adj_seq_array`, of shape (n_nodes, max_prev_node), defined as 

        - `adj_seq_array[i, j] = 1` if j <= i and the node (i+1) is linked to the node (i+1)-(j+1)
        - `adj_seq_array[i, j] = 0` otherwise

    i.e., adj_seq_array[i] is a sequence of 0 and 1, specifying for the node i+1, if the 
    previous nodes (up to `max_prev_node`) are connceted (1) or not (0) to it.

    This function accounts for at maximum the `max_n_nodes` first rows of the ajacency 
    matrix, i.e. truncates the array `adj_seq_array` after the `max_n_nodes-1` first rows
    (if needed).

    Parameters
    ----------
    adj_mat_csr : scipy.sparse.csr_array
        adjacency matrix in csr format
    
    max_n_nodes : int, optional
        maximal number of nodes in the graph (rows of the adjacency matrix `adj_mat_csr`)
        taken into account;
        by default (`None`): `max_n_nodes` is set to n_nodes, where n_nodes is the 
        order of the adjacency matrix `adj_mat_csr` (i.e. all nodes of the graph are
        taken into account)
    
    max_prev_node : int, optional
        maximal number of nodes to look back;
        by default (`None`): `max_prev_node` is set min(n_nodes, `max_n_nodes`)-1 where 
        n_nodes is the order of the adjacency matrix `adj_mat_csr`
        
    Returns
    -------
    adj_seq_array : 2d numpy array of shape (n-1, max_prev_node) of 0 and 1
        encoded ajacency matrix (see above), with n = min(n_nodes, max_n_nodes)

    Notes
    -----
    Special case:
        graph with exactly one node 
        <-> 1 x 1 adjacency matrix [[0.]]
        <-> empty encoded adjacency matrix (empty array)
    """
    if max_n_nodes is None:
        n = adj_mat_csr.shape[0]
    else:
        n = min(adj_mat_csr.shape[0], max_n_nodes)

    if max_prev_node is None:
        max_prev_node = n-1

    # Initialization of adj_seq_array
    adj_seq_array = np.zeros((n-1, max_prev_node), dtype=int)

    # Compute each row of adj_seq_array
    for i in range(1, n):
        jind = adj_mat_csr.indices[adj_mat_csr.indptr[i]:adj_mat_csr.indptr[i+1]]
        jind = jind[np.all((jind < i, jind >= i-max_prev_node), axis=0)]
        adj_seq_array[i-1, i-1-jind] = 1

    # # Equivalent
    # adj_mat_csr_tril = scipy.sparse.tril(adj_mat_csr, k=-1, format='coo')
    # if n < adj_mat_csr.shape[0]:
    #     ind = np.all((adj_mat_csr_tril.row - adj_mat_csr_tril.col <= max_prev_node,
    #                   adj_mat_csr_tril.row < n), axis=0)
    # else:
    #     ind = adj_mat_csr_tril.row - adj_mat_csr_tril.col <= max_prev_node
    # adj_seq_array[adj_mat_csr_tril.row[ind] - 1, adj_mat_csr_tril.row[ind] - adj_mat_csr_tril.col[ind] - 1] = 1

    return adj_seq_array
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def decode_adj(adj_seq_array):
    """
    Decodes array of adjacency sequences ("inverse" of `encode_adj` function).

    Let adj_seq_array a 2d numpy array of shape (m, k), an encoded adjacency
    matrix, the result (output) of the function `encode_adj`, i.e. where       

        - adj_seq_array[i, j] = 0 or 1, with 1 only if j <= i,

    with adj_seq_array[i, j] = 1 meaning that the node (i+1) is linked to the 
    node (i+1)-(j+1).
    This function computes the adjacency matrix in csr format of the graph with
    m+1 nodes from adj_seq_array.

    Parameters
    ----------
    adj_seq_array : 2d numpy array of shape (m, k) of 0 and 1
        encoded adjacency matrix of a graph (see above)
        
    Returns
    -------
    adj_mat_csr : scipy.sparse.csr_array
        adjacency matrix in csr format of the graph corresponding to 
        `adj_seq_array` (see above)

    Notes
    -----
    Special case:
        graph with exactly one node 
        <-> 1 x 1 adjacency matrix [[0.]] 
        <-> empty encoded adjacency matrix (empty array)
    """
    n = adj_seq_array.shape[0] + 1
    i_arr, j_arr = np.nonzero(adj_seq_array)
    j_arr = i_arr - j_arr
    i_arr = i_arr + 1
    i_coo = np.hstack((i_arr, j_arr))
    j_coo = np.hstack((j_arr, i_arr))
    adj_mat_csr = scipy.sparse.coo_array((np.ones(i_coo.shape), (i_coo, j_coo)), shape=(n, n)).tocsr()

    return adj_mat_csr
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Data set to be used with data loader:
#   torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
class Graph_sequence_sampler_data_set(object):
    """
    Class defining a data set from a list of graphs.
    The data set delivers (by `__getitem__`) item `(x, n_nodes)`, where    
    
    - x : 2d tensor of shape `(max_n_nodes, max_prev_node)`
        containing the encoded ajacency matrix of a graph (with some nodes
        numbering)
    - n_nodes : int
        number of nodes taking into account in the encoding of the graph \
        (`n_nodes <= max_n_nodes`), i.e.:
            - the first `n_nodes-1` rows of `x` contain the encoded adjacency \
            matrix
            - the next rows of `x` are filled with zeros (at least one row!)

    Parameters `max_n_nodes` and `max_prev_node` are set by the constructor
    and can be automatically computed (see `__init__`).

    Parameters
    ----------
    G_list : list of `networkx.Graph`
        list of graph
    
    G_nsample : sequence of ints (>=1)
        sequence of same length as `G_list`, of ints >= 1, 
        number of times that each graph in `G_list` is sampled, i.e.
        G_list[i] will be sampled G_nsample[i] times;
        hence, the length of "data set" is the cumulative sum of `G_nsample`
    
    use_bfs : bool, default: `True`
        - if `True`: BFS (breadth-first-search) is used before encoding \
        adjacency matrices
        - if `False`: BFS is not used before encoding adjacency matrices
    
    max_n_nodes : int, optional
        maximal number of nodes taken into account (in a graph);
        by default (`None`): `max_n_nodes` is set to the maximum of the number 
        of nodes of graphs in `G_list`
    
    max_prev_node : int, optional
        maximal number of previous nodes used for encoding adjacency matrices;
        by default (`None`): `max_prev_node` is calculated using the method
        `calc_max_prev_node`
    
    calc_max_prev_node_kwargs : dict, optional
        keyword arguments to be passed to method `calc_max_prev_node` (used if
        `max_prev_node=None`)

    Notes
    -----
    - if `max_prev_node` is not specified, it is computed by the method \
    `calc_max_prev_node`, whose keyword arguments (dict.) can be passed \
    through the parameter `calc_max_prev_node_kwargs`; for reproducibility, a \
    `seed` can be specified in `calc_max_prev_node_kwargs`
    - methods `__len__` and `__getitem__` must be defined, so that instanciated \
    data set can be used with data loader from `pytorch` \
    (`torch.utils.data.DataLoader`)
    - before using a data loader, use `torch.random.manual_seed()` to ensure \
    reproducibility of batches delivered by the data loader (if needed)

    """
    def __init__(self, G_list, G_nsample, use_bfs=True, pos_attr=None, max_n_nodes=None, max_prev_node=None, calc_max_prev_node_kwargs=None):
        """Constructor method.
        """
        # # List of adjacency matrix (in csr format) of each graph
        # self.G_adj_mat_list = [networkx.adjacency_matrix(G) for G in G_list]

        # List of graphs
        self.G_list = G_list

        # List of number of nodes of each graph
        self.G_n_nodes_list = [G.number_of_nodes() for G in G_list]

        # List of indices of sampled graph
        # - G_nsample[i] times i, for i 0, 1, ... len(G_list)-1
        self.G_index_list = np.repeat(range(len(G_nsample)), G_nsample)

        # Length of the data set
        self.len = len(self.G_index_list)

        # Use BFS sequence
        self.use_bfs = use_bfs

        # Node attribute name for spatial positions (enables spatially-ordered BFS)
        self.pos_attr = pos_attr

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
        # Select adjacency matrix and number of nodes
        # adj_mat_csr = np.copy(self.G_adj_mat_list[G_ind])
        # n_nodes = self.G_n_nodes_list[G_ind]
        
        # Select the graph 
        G_ind = self.G_index_list[idx]
        G = self.G_list[G_ind].copy()
        
        # Compute adjacency matrix (starting by reordering nodes randomly)
        # - use only random generator from torch -> reproducibility is then
        #   guaranteed by setting `torch.random.manual_seed()`
        # seq = np.random.permutation(G.number_of_nodes()) # based on numpy
        seq = torch.randperm(G.number_of_nodes()).numpy()
        adj_mat_csr = networkx.adjacency_matrix(G, seq)
        if self.use_bfs:
            if self.pos_attr is not None:
                node_positions = np.array(list(networkx.get_node_attributes(G, self.pos_attr).values()))[seq]
                G = networkx.from_scipy_sparse_array(adj_mat_csr)
                seq = get_spatial_bfs_sequence(G, node_positions)
            else:
                G = networkx.from_scipy_sparse_array(adj_mat_csr)
                seq = get_bfs_sequence(G, 0)
            adj_mat_csr = networkx.adjacency_matrix(G, seq)

        # Encode adjacency matrix
        adj_seq_array = encode_adj(adj_mat_csr, max_n_nodes=self.max_n_nodes, max_prev_node=self.max_prev_node)
        # -> adj_seq_array of shape (n-1, self.max_prev_node), with n <= self.max_n_nodes

        # Define x and n_nodes (to be delivered as item)
        x = torch.zeros((self.max_n_nodes, self.max_prev_node))#, dtype=torch.float32)
        x[0:adj_seq_array.shape[0], :] = torch.from_numpy(adj_seq_array)
        # n_nodes: number of nodes taken into account in the encoding
        n_nodes = adj_seq_array.shape[0] + 1
 
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
# RNN model for graph generation
# ------------------------------
# Model to generate graphs (once model is trained) by iteratively adding one 
# node and the edges (connection) to the already existing nodes.

# The model is constituted of two imbricated RNN models ((instances of) class 
# `RNN_model`):
# - RNN_G : RNN model at graph level (global state of the graph): 
# each "time step" consists of a sequence of 0 or 1 describing the 
# connection (by an edge) from one node to the previous ones.
# - RNN_E : RNN model at edge level (state of connection of one node to 
# the previous ones):
# each "time step" consists of a value 0 or 1 describing the connection
# (by an edge) from one given node to the previous ones.

# These two RNN models are imbricated as illustrated below.

# Using the suffix "_E" resp. "_G" to refer to RNN_E resp. RNN_G, let:
# - T_E   : number of time steps for RNN_E
# - x_E[i]: input for RNN_E for time step i, for i = 0, ..., T_E-1
# - h_E[i]: hidden variable for time step i, for i = 0, ..., T_E-1
#           h_E[T]: hidden variable after time step T_E-1;
# - y_E[i]: output for RNN_E for time step i, for i = 0, ..., T_E-1

# Note that the number of layers are not shown for sake of simplicity,
# but RNN_E may have multiple layer, i.e. the hidden variable h_E[i] is 
# more explicitly h_E[i, layer_index].

# Similar notations for RNN_G.

# Architecture of the model (RNN_G and RNN_E imbricated)
# ------------------------------------------------------
# - connection with symbol "||" and "==":       show the link from RNN_G to RNN_E
# - connection with symbol "v, ^" and ">>, <<": show the link from RNN_E to RNN_G
#
#                        y_E[0] --+                  y_E[1] --+                 y_E[T_E-1]  >> y_E[*] >>+
#                         ^       |                   ^      ...                 ^                      v
#                         |       |                   |                          |                      v
#                    +---------+  |              +---------+                +---------+                 v
#    ++== h_E[0] --> |  RNN_E  |----- h_E[1] --> |  RNN_E  | ----- ...  --> |  RNN_E  | --> h_E[T_E]    v
#    ||              +---------+  |              +---------+                +---------+                 v
#    ||                   ^       |                   ^                          ^                      v
#    ||                   |       |                   |                          |                      v
#    ||                  x_E[0]   |                  x_E[1]                     x_E[T_E-1]              v
#    ||               =[1] (START)|                   ^                          ^                      v
#    ||                           |                   |                          |                      v
#    ||                           +-------------------+                    ... --+                      v
#    ||                                                                                                 v
#    ++=========================++      +<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<+
#                               ||      v                                                                
#                              y_G[t]   v                  ...                                           
#                               ^       v                   ^                                            
#                               |       v                   |                                            
#                          +---------+  v                +---------+                                     
#        ... -- h_G[t] --> |  RNN_G  |----- h_G[t+1] --> |  RNN_G  | --> ...                              
#                          +---------+  v                +---------+                                     
#                               ^       v                     ^                                           
#                               |       v                     |                                           
#                              x_G[t]   v                    x_G[t+1]                                     
#                                       v                     ^                                           
#                                       v                     ^                                           
#                                       +>>>>>>>>>>>>>>>>>>>>>+                                           
# With:
# - RNN_G.input size (size of x_G): max_prev_node, maximal number of previous 
# nodes to look back (in the graph)
# - RNN_E.input_size (size of x_E): 1
# - RNN_E.output_size (size of y_E): 1
# - link from RNN_G to RNN_E ("||" and "==" symbols):
# output of RNN_G -> starting hidden state (of layer 0) of RNN_E, i.e.
# RNN_E.hidden_size = RNN_G.output_size
# - link from RNN_E to RNN_G ("v, ^" and ">>, <<" symbols):
# all outputs of RNN_E form a sequence y_E[*] --> input for next time 
# step of RNN_G

# Generating procedure in details
# -------------------------------
# Considering x_G[t] a sequence of length max_prev_node, we have:
# - step 0: 
#     - start with a graph of one node: node 0
#     - x_G[0] = [1, 1, ..., 1] = SOS (Start Of Sequence), START of RNN_G
#     - RNN_G (one step) gives y_G[0],
#     - x_E[0] = [1] length 1, START of RNN_E
#     - RNN_E is run for one step to get y_E[0], where:
#         y_E[0] : (length 1) probability to have a connection with node 0 
#         for the new next node in the graph
#     - sample wrt y_E[0] to determine the connection for the new next node:
#         y_E[0] replaced by 0 or 1 (according to sample)
# - step 1: 
#     - x_G[1] = [y_E[0], 0, 0, ..., 0]:
#         - if X_G is the null vector, i.e. EOS (End Of Sequence), the generation
#         of the graph stopped
#         - otherwise a new node, node 1, is added to the graph with the
#         connection described by x_G[1]: this gives a graph of 2 nodes
#     - RNN_G (one step) gives y_G[1],
#     - x_E[0] = [1] length 1, START of RNN_E
#     - RNN_E is run for two steps to get y_E[0], y_E[1], where, for each i:
#         - y_E[i] : (length 1) probability to have a connection with node i
#         for the new next node in the graph
#         - sample wrt y_E[i] to determine the connections for the new next node:
#         y_E[i] replaced by 0 or 1 (according to sample) before feeding next 
#         step of RNN_E
# ...
# - step t: 
#     - x_G[t] = sequence of length max_prev_node with 
#         x_G[t][i] = y_E[i], for 0 <= i < min(t, max_prev_node), and 
#         x_G[t][i] = 0, for t <= i < max_prev_node (if any)
#         - if X_G is the null vector, i.e. EOS (End Of Sequence), the generation
#         of the graph stopped
#         - otherwise a new node, node t, is added to the graph with the
#         connection described by x_G[t]: this gives a graph of t+1 nodes
#     - RNN_G (one step) gives y_G[t],
#     - x_E[0] = [1] length 1, START of RNN_E
#     - RNN_E is run for min(t+1, max_prev_node) steps to get
#         y_E[i], for 0 <= i < min(t+1, max_prev_node) where:
#         - y_E[i] : (length 1) probability to have a connection with node i
#         for the new next node in the graph
#         - sample wrt y_E[i] to determine the connections for the new next node:
#         y_E[i] replaced by 0 or 1 (according to sample) before feeding next 
#         step of RNN_E
# ...
# The procedure stops as soon as EOS is obtained.
#
# Training procedure in details
# -----------------------------
# The procedure is similar with the following modifications:
# - the links feeding the inputs of RNN_E and RNN_G are not used: 
# x_E[t] and x_G[t] at any time step t are provided by the data set:
#     - for t >= 0: x_G[t] is the sequence of connection (presence of edge) 
#     from node t of the graph to the previous nodes: 
#         x_G[t][i] = 1 if 0 <= i < min(t, max_prev_node) and an edge exists 
#         between node t and node t-(i+1), and x_G[t][i] = 0 otherwise
#     - x_G[0] is set to SOS (Start Of Sequence): x_G[0][i] = 1 for all i
#     - x_E[0] is set to 1
# - the loss is computed using binary cross-entropy between the output y_E of 
# RNN_E and the data provided by the data set (i.e. x_G at next time step)
# =============================================================================
# ------------------------------------------------------------------------------
def check_rnn_model_graph_gen(rnn_G, rnn_E, data_set=None):
    """
    Checks a RNN model for graph generation.

    This function does some checks about the settings of the two imbricated RNN 
    models ((instances of) class :class:`RNN_model`).
    
    Parameters
    ----------
    rnn_G : class :class:`RNN_model`
        network, RNN model at graph level (global state of the graph)
    
    rnn_E : class :class:`RNN_model`
        network, RNN model at edge level (state of connection of one node to 
        the previous ones)
    
    data_set : (instance of) class `Graph_sequence_sampler_data_set`, optional
        data set, if given, additional check is done
    
    Returns
    -------
    ok : bool
        - `ok=True`: if no inconsistency has been found
        - `ok=False`: if an inconsistency has been found
    
    mes : list of str
        list of messages summarizing the issues found (empty list if `ok=True`)
    """
    ok = True
    mes = []

    if rnn_E.hidden_size != rnn_G.output_size:
        ok = False
        mes.append('Issue: rnn_E.hidden_size != rnn_G.output_size')

    if rnn_E.input_size != 1:
        ok = False
        mes.append('Issue: rnn_E.input_size != 1')

    if rnn_E.output_size != 1:
        ok = False
        mes.append('Issue: rnn_E.output_size != 1')
    if data_set is not None:
        if rnn_G.input_size != data_set.max_prev_node:
            ok = False
            mes.append('Issue: rnn_G.input_size != data_set.max_prev_node')
    
    return ok, mes
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def train_rnn_model_graph_gen(
        rnn_G,
        rnn_E,
        data_loader,
        optimizer_G,
        optimizer_E,
        lr_scheduler_G=None,
        lr_scheduler_E=None,
        return_lr=False,
        return_loss=True,
        num_epochs=10,
        print_epoch=1,
        patience=None,
        min_delta=1e-5,
        device=torch.device('cpu')):
    """
    Trains a RNN model for graph generation.

    The model is constituted of two imbricated RNN models ((instances of) class 
    :class:`RNN_model`):
    
        - `rnn_G` : RNN model at graph level (global state of the graph): \
        each "time step" consists of a sequence of 0 or 1 describing the \
        connection (by an edge) from one node to the previous ones
        - `rnn_E` : RNN model at edge level (state of connection of one node to \
        the previous ones): \
        each "time step" consists of a value 0 or 1 describing the connection \
        (by an edge) from one given node to the previous ones

    Architecture of the model (RNN_G and RNN_E imbricated)

        - connection with symbol "||" and "==":       show the link from RNN_G to RNN_E
        - connection with symbol "v, ^" and ">>, <<": show the link from RNN_E to RNN_G

    ::
    
                           y_E[0] --+                  y_E[1] --+                 y_E[T_E-1]  >> y_E[*] >>+
                            ^       |                   ^      ...                 ^                      v
                            |       |                   |                          |                      v
                       +---------+  |              +---------+                +---------+                 v
       ++== h_E[0] --> |  RNN_E  |----- h_E[1] --> |  RNN_E  | ----- ...  --> |  RNN_E  | --> h_E[T_E]    v
       ||              +---------+  |              +---------+                +---------+                 v
       ||                   ^       |                   ^                          ^                      v
       ||                   |       |                   |                          |                      v
       ||                  x_E[0]   |                  x_E[1]                     x_E[T_E-1]              v
       ||               =[1] (START)|                   ^                          ^                      v
       ||                           |                   |                          |                      v
       ||                           +-------------------+                    ... --+                      v
       ||                                                                                                 v
       ++=========================++      +<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<+
                                  ||      v                                                                
                                 y_G[t]   v                  ...                                           
                                  ^       v                   ^                                            
                                  |       v                   |                                            
                             +---------+  v                +---------+                                     
           ... -- h_G[t] --> |  RNN_G  |----- h_G[t+1] --> |  RNN_G  | --> ...                              
                             +---------+  v                +---------+                                     
                                  ^       v                     ^                                           
                                  |       v                     |                                           
                                 x_G[t]   v                    x_G[t+1]                                     
                                          v                     ^                                           
                                          v                     ^                                           
                                          +>>>>>>>>>>>>>>>>>>>>>+                                           

    Training procedure

        - the links feeding the inputs of RNN_E and RNN_G are not used, \
        x_E[t] and x_G[t] at any time step t are provided by the data set
            - for t >= 0: x_G[t] is the sequence of connection (presence of edge) \
            from node t of the graph to the previous nodes: \
                x_G[t][i] = 1 if 0 <= i < min(t, max_prev_node) and an edge exists \
                between node t and node t-(i+1), and x_G[t][i] = 0 otherwise
            - x_G[0] is set to SOS (Start Of Sequence): x_G[0][i] = 1 for all i
            - x_E[0] is set to 1
        - the loss is computed using binary cross-entropy between the output y_E of \
        RNN_E and the data provided by the data set (i.e. x_G at next time step)

    Parameters
    ----------
    rnn_G : class :class:`RNN_model`
        network, RNN model at graph level (global state of the graph)
    
    rnn_E : class :class:`RNN_model`
        network, RNN model at edge level (state of connection of one node to 
        the previous ones)
    
    data_loader : data loader
        yielding mini-batches (x, n_nodes) from a data set, using 
        `batch_first=True`;
        with:

            - B: current batch size
            - N: data_set.max_n_nodes
            - P: data_set.max_prev_node

            - x: tensor of shape (B, N, P)
                x[k]: k-th input of the mini-batch (encoded adjacency matrix)
            - n_nodes: tensor of shape (B, )
                n_nodes[k]: number of nodes taken into account in the encoding
                `x[k]` (encoded adjacency matrix in the first n_nodes[k]-1 of
                `x[k]`)

    optimizer_G, optimizer_E : optimizer
        updater, torch optimizer for network rnn_G, rnn_E (resp.), e.g. 
        `torch.optim.SGD(net.parameters(), weight_decay=0.001, lr=0.03)`
    
    lr_scheduler_G, lr_scheduler_E : scheduler, optional
        learning rate scheduler for network rnn_G, rnn_E (resp.), e.g. 
        `torch.optim.lr_scheduler.*`
    
    return_lr : bool, default: `False`
        if `True`: sequence of learning rates for network rnn_G, rnn_E (resp.)
        are returned
    
    return_loss : bool, default: `True`
        if `True`: sequence of losses are returned
    
    num_epochs : int, default: 10
        number of epochs
    
    print_epoch : int, default: 10
        result of every `print_epoch` epoch is displayed in stdout, if
        `print_epoch > 0`

    patience : int, optional
        number of epochs with no improvement after which training stops early;
        by default (`None`): early stopping is disabled and training always
        runs for `num_epochs` epochs

    min_delta : float, default: 1e-5
        minimum decrease in loss that counts as an improvement for patience

    device : torch device, default: torch.device('cpu')
        device on which the network is trained

    Returns
    -------
    loss : list, optional
        returned if `return_loss=True`, training loss of every epoch,
        list of floats of length `num_epochs` (or fewer if early stopping fires)
    
    lr_used_G : list, optional
        returned if `return_lr=True`, learning rate for rnn_G used at 
        each epoch, list of floats of length `num_epochs`        
    
    lr_used_E : list, optional
        returned if `return_lr=True`, learning rate for rnn_E used at 
        each epoch, list of floats of length `num_epochs`        
    """
    fname = 'train_rnn_model_graph_gen'

    print('*** Training on', device, '***')

    # Copy models to device
    rnn_G.to(device)
    rnn_E.to(device)

    # Set models in training mode
    rnn_G.train()
    rnn_E.train()

    # Initialize list for loss
    if return_loss:
        loss = []

    # Initialize list for lr
    if return_lr:
        lr_used_G = []
        lr_used_E = []

    # Some initializations
    if return_loss:
        loss_len = 0      # reset loss length
        loss_epoch = 0.0  # reset loss of one epoch

    # Early stopping state
    if patience is not None:
        best_loss = float('inf')
        patience_counter = 0

    # Train the model
    for epoch in range(num_epochs):
        # rnn_G.train()
        # rnn_E.train()
        # for batch_idx, (x, n_nodes) in enumerate(data_loader):
        for x, n_nodes in data_loader:
            # Let 
            #   data_set: data set loaded by data_loader
            #   B: current batch size
            #   N: data_set.max_n_nodes
            #   P: data_set.max_prev_node
            # then
            #   x       : tensor of shape (B, N, P)
            #   n_nodes : tensor of shape (B, ):
            #      number of nodes in the underlying graphs (encoded in x), i.e.
            #         x[i, 0:n_nodes[i]-1, :]
            #      encoded adjacency matrix for the i-th element of the mini-batch,
            #      and:
            #         x[i, n_nodes[i]-1:, :] = 0 (zeros)

            # Cut x tensor along dim 1, wrt. maximum of n_nodes
            n_nodes_max = max(n_nodes)
            x = x[:, 0:n_nodes_max, :]
            # rnn_G: input (x_G), output (y_G), packing lengths (pack_lens_G)
            # ---------------------------------------------------------------
            # Sort x, n_nodes, such that n_nodes is in decreasing order
            # (i.e. from the largest to the smallest batch element)
            # -> y_G: tensor containing x reordered
            # -> x_G: tensor obtained by shifting y_G of one position along the dim 1,
            #         and inserting ones in the first position (along dim 1)
            n_nodes_G, sort_index = torch.sort(n_nodes, dim=0, descending=True)
            y_G = torch.index_select(x, dim=0, index=sort_index).to(device)
            x_G = torch.cat((torch.ones((y_G.size(0), 1, y_G.size(2)), device=device), y_G[:,:n_nodes_max-1,:]), dim=1)
            # --- or: 
            # x_G = torch.ones_like(y_G, device=device)
            # x_G[:,1:n_nodes_max,:] = y_G[:, :n_nodes_max-1, :]
            # ---            
            # -> x_G, y_G of shape (B, N', P), with N' = max(n_nodes)

            # Set length of each element of the mini-batch for rnn_G
            pack_lens_G = n_nodes_G
            
            # rnn_G: hidden varible (state_G)
            # --------------------------------
            state_G = rnn_G.init_state(batch_size=x_G.size(0), device=device) # shape (L, B, H) where L = rnn_G.num_layers, H = rnn_G.hiddensize
                                                                              # (state_G is a 2-tuple if rnn_G is LSTM)

            # rnn_E: input (x_E), output (y_E), packing lengths (pack_lens_E)
            # ---------------------------------------------------------------
            # Pack y_G, wrt to lengths pack_lens_G (already sorted in descending order -> `enforce_sorted=False` below)
            y_G_packed = torch.nn.utils.rnn.pack_padded_sequence(y_G, pack_lens_G, batch_first=True, enforce_sorted=False)
            # -> y_G_packed.data: tensor of shape (sum(pack_lens_G), P)
            # -> y_G_packed.batch_sizes: tensor of dim 1 with sum(y_G_packed.batch_sizes) = sum(pack_lens_G), and:
            #    - first k = y_G_packed.batch_sizes[0] indices (along dim=0) in y_G_packed.data are the 1st entry (index 0) of the k first elements of the mini-batch for rnn_G
            #    - next  k = y_G_packed.batch_sizes[1] indices (along dim=0) in y_G_packed.data are the 2nd entry (index 1) of the k next elements of the mini-batch for rnn_G
            #    - etc.
            # i.e.:
            #    - the k = y_G_packed.batch_sizes[0] first rows in y_G_packed.data correspond to sequences of length 1 (more precisely min(1, P))
            #    - the k = y_G_packed.batch_sizes[1] next  rows in y_G_packed.data correspond to sequences of length 2 (more precisely min(2, P))
            #    ...
            
            # Set 
            #   y_E        : y_G_packed.data
            #   pack_lens_E: lengths of sequences in each row of y_E 
            # note: y_G_packed.data on specified device `device`, whereas unique_lens, y_G_packed.batch_sizes on cpu
            y_E = y_G_packed.data # shape (B_E, P) (with B_E = sum(pack_lens_G))
            unique_lens = torch.minimum(torch.arange(1, len(y_G_packed.batch_sizes)+1), torch.full((len(y_G_packed.batch_sizes), ), y_G_packed.data.size(1)))
            pack_lens_E = torch.repeat_interleave(unique_lens, y_G_packed.batch_sizes)
            # -> pack_lens_E of length B_E

            # Reverse order of the sequences (y_E along dim=0), so that the sequences corresponding to the rows are of decreasing lengths
            y_E_rev_index = torch.arange(y_E.size(0)-1, -1, -1, device=device)
            y_E = torch.index_select(y_E, dim=0, index=y_E_rev_index)
            pack_lens_E = torch.index_select(pack_lens_E, dim=0, index=y_E_rev_index.to('cpu')) # note: pack_lens_E and index tensor on 'cpu'

            # Add a dimension (of size 1) at the end to y_E
            # and set x_E: prepend ones to y_E along dim 1 (and cut last entry along dim 1)
            y_E = y_E.unsqueeze(-1)
            x_E = torch.cat((torch.ones((y_E.size(0), 1, 1), device=device), y_E[:, :-1, :]), dim=1)
            # -> x_E, y_E of shape (B_E, P, 1)
            #    B_E = sum(pack_lens_G) = x_E.size(0) = y_E.size(0): is the current batch size for rnn_E
            
            # rnn_E: hidden varible (state_E)
            # Set further: output of rnn_G is used as hidden state at layer 0

            # Run rnn_G
            # ---------
            output_G, state_G = rnn_G(x_G, state_G, pack=True, pack_lens=pack_lens_G) # or rnn_G.forward(...)
            # -> output_G of shape (B, N', O_G), with O_G = rnn_G.output_size

            # rnn_E: hidden varible (state_E)
            # -------------------------------
            # Set output of rnn_G as hidden state at layer 0
            #
            # Get packed data of output_G
            output_G_packed_data = torch.nn.utils.rnn.pack_padded_sequence(output_G, pack_lens_G, batch_first=True, enforce_sorted=False).data 
            # -> output_G_packed_data of shape (B_E, O_G), with 
            #    B_E = sum(pack_lens_G) = x_E.size(0) = y_E.size(0)
            #    O_G = rnn_G.output_size = rnn_E.hidden_size = H_E
            #
            # Reverse order (according to what has been done for y_E)
            output_G_packed_data = torch.index_select(output_G_packed_data, dim=0, index=y_E_rev_index)

            state_E = rnn_E.init_state(batch_size=x_E.size(0), device=device) # shape (L_E, B_E, H_E) where L_E = rnn_E.num_layers, H_E = rnn_E.hiddensize
                                                                              # (state_E is a 2-tuple if rnn_E is LSTM)

            if isinstance(state_E, tuple):
                state_E[0][0, ...] = output_G_packed_data
            else:
                state_E[0, ...] = output_G_packed_data

            # # --- Equivalent ---
            # if isinstance(rnn_E.rnn, torch.nn.LSTM):
            #     state_E = (torch.cat((output_G_packed_data.unsqueeze(0), torch.zeros(rnn_E.num_layers-1, output_G_packed_data.size(0), output_G_packed_data.size(1))), dim=0),
            #                torch.zeros(rnn_E.num_layers, output_G_packed_data.size(0), output_G_packed_data.size(1)))
            # else:
            #     state_E = torch.cat((output_G_packed_data.unsqueeze(0), torch.zeros(rnn_E.num_layers-1, output_G_packed_data.size(0), output_G_packed_data.size(1))), dim=0)
            # # ------------------

            # Run rnn_E
            # ---------
            output_E, state_E = rnn_E(x_E, state_E, pack=True, pack_lens=pack_lens_E) # or rnn_E.forward(...)
            # -> output_E of shape (B_E, P, 1), with 1 = O_E = rnn_E.output_size=1

            # Get packed data of output_E and y_E
            output_E_packed_data = torch.nn.utils.rnn.pack_padded_sequence(output_E, pack_lens_E, batch_first=True).data 
            y_E_packed_data      = torch.nn.utils.rnn.pack_padded_sequence(     y_E, pack_lens_E, batch_first=True).data 
            # -> output_E_packed_data, y_E_packed_data of shape (sum(pack_lens_E), 1), with 1 = O_E = rnn_E.output_size=1

            # Activate output of rnn_E: sigmoid (interpretetion as probabilities)
            y_E_hat_packed_data = torch.nn.functional.sigmoid(output_E_packed_data)

            # Compute loss
            # ------------
            # Binary cross-entropy between y_E_hat_packed_data and y_E_packed_data is used
            loss_mini_batch = torch.nn.functional.binary_cross_entropy(y_E_hat_packed_data, y_E_packed_data)

            # Update model
            # ------------
            # Reset gradient in optimized tensors
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            # Compute gradient of loss wrt. to model parameters
            loss_mini_batch.backward()
            # Update for the current mini-batch (one training step)
            optimizer_G.step()
            optimizer_E.step()
            
            if return_loss:
                # Number of entries contributing in loss_mini_batch, i.e. y_E_packed_data.size(0) = sum(pack_lens_E)
                #    note: y_E_packed_data.size(1) = y_E.size(2) = 1
                loss_mini_batch_len = y_E_packed_data.size(0)
                # Update loss for entire epoch (for saving further)
                loss_len += loss_mini_batch_len
                loss_epoch += float(loss_mini_batch)*loss_mini_batch_len

        if return_loss:
            loss.append(loss_epoch/loss_len)
            loss_len = 0      # reset loss length
            loss_epoch = 0.0  # reset loss of one epoch

        if return_lr:
            lr_used_G.append(optimizer_G.param_groups[0]['lr'])
            lr_used_E.append(optimizer_E.param_groups[0]['lr'])
        if lr_scheduler_G:
            # lr_used_G.append(lr_scheduler_G.get_last_lr()[0])
            lr_scheduler_G.step()
        if lr_scheduler_E:
            # lr_used_E.append(lr_scheduler_E.get_last_lr()[0])
            lr_scheduler_E.step()

        if print_epoch > 0 and epoch % print_epoch == 0:
            # Print result of current epoch
            s = f'epoch {epoch+1} of {num_epochs}'
            if return_loss:
                s = s + f', loss : {loss[-1]:.6f}'
            print(s)

        # Early stopping
        if patience is not None and return_loss:
            if loss[-1] < best_loss - min_delta:
                best_loss = loss[-1]
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
                break

    # Set model on cpu
    rnn_G.to(torch.device('cpu'))
    rnn_E.to(torch.device('cpu'))

    out = []
    if return_loss:
        out.append(loss)
    if return_lr:
        out.append(lr_used_G)
        out.append(lr_used_E)
    out = tuple(out)
    if len(out) == 1:
        out = out[0]
    elif len(out) == 0:
        out = None
    return out
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def generate_graph(
        rnn_G,
        rnn_E,
        max_n_nodes,
        n_graph=1,
        force_node1=True,
        return_encoded=False,
        device=None):
    """
    Generates one or several graph(s) using a RNN model for graph generation.

    The model is constituted of two imbricated RNN models ((instances of) class
    :class:`RNN_model`):

        - `rnn_G` : RNN model at graph level (global state of the graph): \
        each "time step" consists of a sequence of 0 or 1 describing the \
        connection (by an edge) from one node to the previous ones
        - `rnn_E` : RNN model at edge level (state of connection of one node to \
        the previous ones): \
        each "time step" consists of a value 0 or 1 describing the connection \
        (by an edge) from one given node to the previous ones

    Architecture of the model (RNN_G and RNN_E imbricated)

        - connection with symbol "||" and "==":       show the link from RNN_G to RNN_E
        - connection with symbol "v, ^" and ">>, <<": show the link from RNN_E to RNN_G
    
    ::
    
                           y_E[0] --+                  y_E[1] --+                 y_E[T_E-1]  >> y_E[*] >>+
                            ^       |                   ^      ...                 ^                      v
                            |       |                   |                          |                      v
                       +---------+  |              +---------+                +---------+                 v
       ++== h_E[0] --> |  RNN_E  |----- h_E[1] --> |  RNN_E  | ----- ...  --> |  RNN_E  | --> h_E[T_E]    v
       ||              +---------+  |              +---------+                +---------+                 v
       ||                   ^       |                   ^                          ^                      v
       ||                   |       |                   |                          |                      v
       ||                  x_E[0]   |                  x_E[1]                     x_E[T_E-1]              v
       ||               =[1] (START)|                   ^                          ^                      v
       ||                           |                   |                          |                      v
       ||                           +-------------------+                    ... --+                      v
       ||                                                                                                 v
       ++=========================++      +<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<+
                                  ||      v                                                                
                                 y_G[t]   v                  ...                                           
                                  ^       v                   ^                                            
                                  |       v                   |                                            
                             +---------+  v                +---------+                                     
           ... -- h_G[t] --> |  RNN_G  |----- h_G[t+1] --> |  RNN_G  | --> ...                              
                             +---------+  v                +---------+                                     
                                  ^       v                     ^                                           
                                  |       v                     |                                           
                                 x_G[t]   v                    x_G[t+1]                                     
                                          v                     ^                                           
                                          v                     ^                                           
                                          +>>>>>>>>>>>>>>>>>>>>>+                                           

    Generating procedure
        - start with a graph of one node: node 0
        - x_G[0] is set to SOS (Start Of Sequence): x_G[0][i] = 1 for all i
        - then, for any time step t-1 (with t>=1):
            - the outputs y_E[i] of RNN_E are interpreted as probabilities \
            (activated by sigmoid)
            - random number (0 or 1) are drawn according to Bernoulli distribution \
            of parameter y_E[i] (i.e. value 1 with probability y_E[i]), and y_E[i] \
            is replaced by the value sampled (0 or 1), before feeding the next step \
            of RNN_E, for each i
            - the vector y_E[*] (binary with sampled values) feeds the next step of \
            RNN_G, x_G[t], with
                - x_G[t][i] = y_E[i], for 0 <= i < min(t, max_prev_node), and \
                x_G[t][i] = 0, for t <= i < max_prev_node (if any)
            - if x_G[t] is the vector with only zeros, i.e. EOS (End Of Sequence), \
            the generation of the graph stopped
            - otherwise a new node, node t, is added to the graph with the \
            connection described by x_G[t]: this gives a graph of t+1 nodes
        - the procedure stops as soon as EOS is obtained or a maximal \
        number of nodes is reached
    
    Parameters
    ----------
    rnn_G : class :class:`RNN_model`
        network, RNN model at graph level (global state of the graph)
    
    rnn_E : class :class:`RNN_model`
        network, RNN model at edge level (state of connection of one node to 
        the previous ones)
    
    max_n_nodes : int
        maximal number of nodes in the generated graph(s)
    
    n_graph : int, default: 1
        number of graph to be generated
    
    force_node1 : bool, default: `True`
        - if `True` : force creation of node 1 with edge between nodes 0 and 1, \
        i.e. at time step t=0, no sampling (y_E[0] is set to 1 by force)
        - if `False`: no forcing for node 1, i.e. \
        at time step t=0, y_E[0] is sampled
    
    return_encoded : bool, default: `False`
        if `True`: the encoded adjacency matrix is returned

    device : torch device, optional
        by default (None): default_device() is used (MPS > CUDA > CPU)

    Returns
    -------
    G_list : list of networkx.Graph object of length `n_graph`
        generated graphs:

        - G_list[k]: k-th graph

    adj_seq_array_list : list of 2d numpy arrays of length `n_graph`, optional
        encoded adjacency matrices:
        
        - adj_seq_array_list[k]: encoded adjacency matrix from which the \
        k-th graph is obtained, 2d numpy array of shape (n-1, max_prev_node) \
        where n is the number of nodes in the graph G_list[k] (with \
        `n <= max_n_nodes`)

    Notes
    -----
    If the EOS is obtained for X_G[1], then the generated graph has exactly
    one node, and the encoded adjacency matrix is empty (size=0)
    """
    fname = 'generate_graph'

    if device is None:
        device = default_device()

    # Copy models to device
    rnn_G.to(device)
    rnn_E.to(device)

    # Set models in evaluation mode
    rnn_G.eval()
    rnn_E.eval()

    # Initialization of adj_seq: tensor to store encoded adjacency matrices of generated graphs
    max_prev_node = rnn_G.input_size
    adj_seq = torch.zeros((n_graph, max_n_nodes-1, max_prev_node))

    # rnn_G: input (x_G)
    # ------------------
    # SOS (Start Of Sequence)
    x_G = torch.ones((n_graph, 1, max_prev_node), device=device) # shape (B=batch_size=n_graph, N=1, rnn_G.input_size=max_prev_node)

    # rnn_G: hidden variable (state_G)
    # --------------------------------
    # Initialize
    state_G = rnn_G.init_state(batch_size=x_G.size(0), device=device) # shape (L, B, H) where L = rnn_G.num_layers, H = rnn_G.hiddensize
                                                                      # (state_G is a 2-tuple if rnn_G is LSTM)
    if force_node1:
        with torch.no_grad():
            output_G, state_G = rnn_G(x_G, state_G) # or rnn_G.forward(...)
        # Set x_G for next step
        x_G = torch.zeros((n_graph, 1, rnn_G.input_size), device=device) # initialization of x_G for next step in rnn_G
        x_G[:, 0, 0] = 1 # force connection to node 0
        # Set 0-th row of encoded adjacency matrices
        adj_seq[:, 0, :] = x_G[:, 0, :]
        i_start = 1
    else:
        i_start = 0

    for i in range(i_start, max_n_nodes-1):
        # Generate i-th row of encoded adjacency matrices
        # -> adj_seq[:, i, 0:min(i+1, max_prev_node)]
        
        # Run rnn_G
        # ---------
        with torch.no_grad():
            output_G, state_G = rnn_G(x_G, state_G) # or rnn_G.forward(...)
            # -> output_G of shape (B, 1, O_G), with O_G = rnn_G.output_size = rnn_E.hidden_size = H_E

        # rnn_E: input (x_E)
        # ------------------
        # SOS (Start Of Sequence)
        x_E = torch.ones((n_graph, 1, 1), device=device) # shape (B=batch_size=n_graph, N=1, rnn_E.input_size=1)

        # rnn_E: hidden variable (state_E)
        # --------------------------------
        # Initialize
        state_E = rnn_E.init_state(batch_size=x_E.size(0), device=device) # shape (L_E, B, H_E) where L_E = rnn_E.num_layers, H_E = rnn_E.hiddensize
                                                                          # (state_E is a 2-tuple if rnn_E is LSTM)
        # Set output of rnn_G as hidden state at layer 0
        if isinstance(state_E, tuple):
            state_E[0][0, ...] = output_G[:, 0, :]
        else:
            state_E[0, ...] = output_G[:, 0, :]

        x_G = torch.zeros((n_graph, 1, rnn_G.input_size), device=device) # initialization of x_G for next step in rnn_G
        for j in range(min(i+1, max_prev_node)):
            # Generate absence or presence of edge (value 0 or 1) between node i+1 and node (i+1)-(j+1)
            # Run rnn_E
            # ---------
            with torch.no_grad():
                output_E, state_E = rnn_E(x_E, state_E) # or rnn_E.forward(...)
                # -> output_E of shape (B, 1, O_E), with O_E = rnn_E.output_size = 1

            # Activate output of rnn_E: sigmoid (interpretetion as probabilities)
            y_E_hat = torch.nn.functional.sigmoid(output_E)

            # Sample according to y_E_hat -> get 0, 1 value in x_E for next step in rnn_E
            x_E = torch.lt(torch.rand(y_E_hat.size(), device=device), y_E_hat).to(torch.float)

            # Update x_G
            x_G[:, 0, j] = x_E[:, 0, 0]

        #print(x_G, torch.sum(x_G), torch.all(x_G[:, 0, :] == 0))
        if torch.all(x_G[:, 0, :] == 0):
            # EOS is obtained for all graphs
            break

        # Set i-th row of encoded adjacency matrices
        adj_seq[:, i, :] = x_G[:, 0, :]
        # # or : 
        # adj_seq[:, i, 0:min(i+1, max_prev_node)] = x_G[:, 0, 0:min(i+1, max_prev_node)]

    # Set model on cpu
    rnn_G.to(torch.device('cpu'))
    rnn_E.to(torch.device('cpu'))

    # Get graphs (and encoded adjacency matrices if required)
    G_list = []
    adj_seq_array_list = []
    for k in range(n_graph):
        # Get encoded adjacency matrix of index k
        adj_seq_array = adj_seq[k].numpy().astype(int) # shape (max_n_nodes-1, max_prev_node)
        # Cut it from the first row filled with zeros (if needed)
        ind_all_zeros = np.where(np.all(adj_seq_array==0, axis=1))[0]
        if len(ind_all_zeros):
            adj_seq_array = adj_seq_array[0:ind_all_zeros[0], :]

        if return_encoded:
            adj_seq_array_list.append(adj_seq_array)

        # Decode, and build the corresponding graph
        adj_mat_csr = decode_adj(adj_seq_array)
        G = networkx.from_scipy_sparse_array(adj_mat_csr)
        G_list.append(G)

    out = [G_list]
    if return_encoded:
        out.append(adj_seq_array_list)
    out = tuple(out)
    if len(out) == 1:
        out = out[0]
    return out
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def generate_graph_min_n_nodes(
        rnn_G,
        rnn_E,
        min_n_nodes,
        max_n_nodes,
        n_graph=1,
        force_node1=True,
        return_encoded=False,
        device=None):
    """
    Generates one or several graph(s) using a RNN model for graph generation.

    The function `generate_graph` is iteratively called (without the argument 
    `min_n_nodes`) to replace the generated graphs having less than `min_n_nodes`
    nodes. See this function for details.
    
    Parameters
    ----------
    min_n_nodes : int
        minimal number of nodes in each generated graph (new graph are re-generated
        if needed)
    
    Notes
    -----
    See function `generate_graph` for other parameters and returns
    """
    fname = 'generate_graph_min_n_nodes'

    if device is None:
        device = default_device()

    if min_n_nodes > max_n_nodes:
        print('ERROR ({fname}): `min_n_nodes` less than `max_n_nodes`')
        return None

    out = generate_graph(
        rnn_G,
        rnn_E,
        max_n_nodes=max_n_nodes,
        n_graph=n_graph,
        force_node1=force_node1,
        return_encoded=return_encoded,
        device=device)

    if return_encoded:
        G_list, adj_seq_array_list = out
    else:
        G_list = out

    ind_too_small = [G.number_of_nodes() < min_n_nodes for G in G_list]
    while np.any(ind_too_small):
        m = np.sum(ind_too_small)
        out2 = generate_graph(
            rnn_G,
            rnn_E,
            max_n_nodes=max_n_nodes,
            n_graph=m,
            force_node1=force_node1,
            return_encoded=return_encoded,
            device=device)
            
        if return_encoded:
            G_list2, adj_seq_array_list2 = out2
            for i, j in enumerate(np.nonzero(ind_too_small)[0]):
                G_list[j] = G_list2[i]
                adj_seq_array_list[j] = adj_seq_array_list2[i]
        else:
            G_list2 = out2
            for i, j in enumerate(np.nonzero(ind_too_small)[0]):
                G_list[j] = G_list2[i]
        
        ind_too_small = [G.number_of_nodes() < min_n_nodes for G in G_list]

    return out
# ------------------------------------------------------------------------------

# =============================================================================
# Encoding / decoding adjacency matrix and node features
# =============================================================================
# -----------------------------------------------------------------------------
def encode_adj_and_node_features(adj_mat_csr, node_features_array, max_n_nodes=None, max_prev_node=None):
    """
    Encodes an adjacency matrix in csr format of a graph and node features.

    Let n_nodes be the number of nodes in the graph. The "encoded adjacency matrix" 
    consists of a 2d arrray `adj_seq_array`, of shape (n_nodes, max_prev_node), defined as 

        - `adj_seq_array[i, j] = 1` if j <= i and the node (i+1) is linked to the node (i+1)-(j+1)
        - `adj_seq_array[i, j] = 0` otherwise
    
    i.e., adj_seq_array[i] is a sequence of 0 and 1, specifying for the node i+1, if the 
    previous nodes (up to `max_prev_node`) are connceted (1) or not (0) to it.

    Moreover, given the numpy array `node_features_array`, of shape (n_nodes, n_node_features), 
    where `node_features_array[i]` are the features at node i, the "encoded node features 
    differences" consists of a 3d array `node_feature_seq_array`, of shape 
    (n_nodes-1, max_prev_node, n_node_features), defined as

        - `node_features_seq_array[i, j] = node_features_array[i+1, :] - node_features_array[i-j, :]` \
        if `adj_seq_array[i, j] = 1`
        - `node_features_seq_array[i, j] = 0` otherwise

    i.e., `node_features_seq_array[i]` contains the features at node i+1 minus the features
    at the previous nodes (up to max_prev_node) that are connected to it.
        
    This function accounts for at maximum the `max_n_nodes` first rows of the ajacency 
    matrix, i.e. truncates the arrays `adj_seq_array` and `node_features_seq_array` after 
    the `max_n_nodes-1` first rows (if needed).

    Parameters
    ----------
    adj_mat_csr : scipy.sparse.csr_array
        adjacency matrix in csr format
    
    node_features_array : 2d numpy array 
        node features, array of shape (n_nodes, n_node_features), where n_nodes
        is the order of the adjacency matrix `adj_mat_csr`, and n_node_features the number
        of features at each node
    
    max_n_nodes : int, optional
        maximal number of nodes in the graph (rows of the adjacency matrix `adj_mat_csr`)
        taken into account;
        by default (`None`): `max_n_nodes` is set to n_nodes, where n_nodes is the 
        order of the adjacency matrix `adj_mat_csr` (i.e. all nodes of the graph are
        taken into account)
    
    max_prev_node : int, optional
        maximal number of nodes to look back;
        by default (`None`): `max_prev_node` is set min(n_nodes, `max_n_nodes`)-1 where 
        n_nodes is the order of the adjacency matrix `adj_mat_csr`
        
    Returns
    -------
    adj_seq_array : 2d numpy array of shape (n-1, max_prev_node) of 0 and 1
        encoded ajacency matrix (see above), with n = min(n_nodes, max_n_nodes)
    
    node_features_seq_array : 3d numpy array of shape (n-1, max_prev_node, n_node_features)
        encoded node features differences wrt. `adj_seq_array` (see above)

    Notes
    -----
    Special case:
        graph with exactly one node 
        <-> 1 x 1 adjacency matrix [[0.]] 
        <-> empty encoded adjacency matrix and encoded node features differences (empty arrays)
    """
    if max_n_nodes is None:
        n = adj_mat_csr.shape[0]
    else:
        n = min(adj_mat_csr.shape[0], max_n_nodes)

    if max_prev_node is None:
        max_prev_node = n-1

    # Initialization of adj_seq_array and node_features_seq_array
    adj_seq_array           = np.zeros((n-1, max_prev_node), dtype=int)
    node_features_seq_array = np.zeros((n-1, max_prev_node, node_features_array.shape[1]))

    # Compute each row of adj_seq_array and node_features_seq_array
    for i in range(1, n):
        jind = adj_mat_csr.indices[adj_mat_csr.indptr[i]:adj_mat_csr.indptr[i+1]]
        jind = jind[np.all((jind < i, jind >= i-max_prev_node), axis=0)]
        adj_seq_array[i-1, i-1-jind] = 1
        node_features_seq_array[i-1, i-1-jind] = node_features_array[i] - node_features_array[jind]

    return adj_seq_array, node_features_seq_array
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def decode_adj_and_node_features(adj_seq_array, node_features_seq_array, features0=None):
    """
    Decodes arrays of adjacency and node features sequences 
    ("inverse" of `encode_adj_and_node_features` function).

    Let `adj_seq_array` a 2d numpy array of shape (m, k) an encoded adjacency 
    matrix, and `node_features_seq_array` a 3d numpy array of shape 
    (m, k, n_node_features) the encoded node features differences wrt. 
    `adj_seq_array` (with n_node_features the number of features attached to each 
    node), the result (output) of the function `encode_adj_and_node_features`, i.e. 
    where

        - `adj_seq_array[i, j]` = 0 or 1, with 1 only if j <= i,

    with `adj_seq_array[i, j] = 1` meaning that the node (i+1) is linked to the 
    node (i+1)-(j+1), and `node_features_seq_array[i, j]` the features at node i+1 
    minus the features at node (j+1);
    `node_features_seq_array[i, j]` not used if adj_seq_array[i, j] = 0. 
    
    This function computes the adjacency matrix in csr format of the graph with
    m+1 nodes from `adj_seq_array`, and the features at the (m+1) nodes by assuming 
    the features at node 0 equal to `features0` (zeros by default) and by taking,
    for each next node, the mean of the features computed from
    `node_features_seq_array` and the previous connected nodes.

    Parameters
    ----------
    adj_seq_array : 2d numpy array of shape (m, k) of 0 and 1
        encoded adjacency matrix of a graph (see above)
    
    node_features_seq_array : 3d numpy array of shape (m, k, n_node_features) of floats
        encoded node features differences wrt. `adj_seq_array` (see above)
        
    Returns
    -------
    adj_mat_csr : scipy.sparse.csr_array
        adjacency matrix in csr format of the graph corresponding to 
        `adj_seq_array` (see above)
    
    node_features_array : 2d numpy array of floats
        node features, node_features_array[i]: features at node i
    
    features0 : numpy array of shape (n_node_features, ), optional
        features at node 0;
        by default: `features0 = numpy.zeros(n_node_features)` is used 

    Notes
    -----
    Special case:
        graph with exactly one node 
        <-> 1 x 1 adjacency matrix [[0.]] 
        <-> empty encoded adjacency matrix and encoded node features differences (empty arrays)
    """
    n = adj_seq_array.shape[0] + 1
    i_arr, j_arr = np.nonzero(adj_seq_array)
    j_arr = i_arr - j_arr
    i_arr = i_arr + 1
    i_coo = np.hstack((i_arr, j_arr))
    j_coo = np.hstack((j_arr, i_arr))
    adj_mat_csr = scipy.sparse.coo_array((np.ones(i_coo.shape), (i_coo, j_coo)), shape=(n, n)).tocsr()
    
    node_features_array = np.zeros((n, node_features_seq_array.shape[2]))
    if features0 is not None:
        node_features_array[0] = features0
    for i in range(1, n):
        indi = j_arr[i_arr==i]
        if len(indi):
            node_features_array[i] = np.mean(node_features_array[indi] + node_features_seq_array[i-1, i-1-indi], axis=0)
        # else: node_features_array[i] will be np.zeros(node_features_seq_array.shape[2])

    return adj_mat_csr, node_features_array
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Data set to be used with data loader:
#   torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
class Graph_sequence_sampler_data_set_with_node_features(object):
    """
    Class defining a data set from a list of graphs.
    The data set delivers (by `__getitem__`) item `(x, x_nf, n_nodes)`, 
    where

    - x : 2d tensor of shape `(max_n_nodes, max_prev_node)`
        containing the encoded ajacency matrix of a graph (with some nodes 
        numbering)
    - x_nf : 3d tensor of shape `(max_n_nodes, max_prev_node, n_node_features)`
        containing the encoded node features differences wrt. encoded ajacency 
        matrix
    - n_nodes : int
        number of nodes taking into account in the encoding of the graph
        (`n_nodes <= max_n_nodes`), i.e.:
        - the first `n_nodes-1` rows of `x` (resp. `x_nf`) contain the encoded 
        adjacency matrix (resp. encoded node features differences)
        - the next rows of `x` and `x_nf` are filled with zeros (at least one row!)

    Parameters `max_n_nodes` and `max_prev_node` are set by the constructor
    and can be automatically computed (see `__init__`). The number of features 
    attached to each node, `n_node_features`, is automatically determined 
    (from the graph(s) given to the constructor); all nodes of each graph are 
    assumed to have an attribute of same name (`attr`) which is an 1d-array 
    of shape (`n_node_features`, ).

    Parameters
    ----------
    G_list : list of `networkx.Graph`
        list of graph
    
    G_nsample : sequence of ints (>=1)
        sequence of same length as `G_list`, of ints >= 1, 
        number of times that each graph in `G_list` is sampled, i.e.
        G_list[i] will be sampled G_nsample[i] times;
        hence, the length of "data set" is the cumulative sum of `G_nsample`
    
    attr : str, default: 'pos'
        name of the attribute attached to nodes, the attribute is a sequence
        of n_node_features floats; all nodes of every graph in `G_list` are 
        assumed to have this attribute (of same type);
        by default: 'pos' is used (position of the nodes)
    
    use_bfs : bool, default: `True`
        - if `True`: BFS (breadth-first-search) is used before encoding \
        adjacency matrices
        - if `False`: BFS is not used before encoding adjacency matrices
    
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

    Notes
    -----
    - if `max_prev_node` is not specified, it is computed by the method \
    `calc_max_prev_node`, whose keyword arguments (dict.) can be passed \
    through the parameter `calc_max_prev_node_kwargs`; for reproducibility, a \
    `seed` can be specified in `calc_max_prev_node_kwargs`
    - methods `__len__` and `__getitem__` must be defined, so that instanciated \
    data set can be used with data loader from `pytorch` \
    (`torch.utils.data.DataLoader`)
    - before using a data loader, use `torch.random.manual_seed()` to ensure \
    reproducibility of batches delivered by the data loader (if needed)
    """
    def __init__(self, G_list, G_nsample, attr='pos', use_bfs=True, max_n_nodes=None, max_prev_node=None, 
                 node_features_noise_std=None, calc_max_prev_node_kwargs=None):
        """Constructor method.
        """
        # # List of adjacency matrix (in csr format) of each graph 
        # self.G_adj_mat_list = [networkx.adjacency_matrix(G) for G in G_list]

        # List of graphs 
        self.G_list = G_list

        # List of number of nodes of each graph
        self.G_n_nodes_list = [G.number_of_nodes() for G in G_list]

        # List of indices of sampled graph
        # - G_nsample[i] times i, for i 0, 1, ... len(G_list)-1
        self.G_index_list = np.repeat(range(len(G_nsample)), G_nsample)

        # Attribute to be considered
        self.attr = attr

        # Number of features (in the considered attribute)
        self.n_node_features = len(self.G_list[0].nodes[0][self.attr])
        # note: all nodes of every graph in `G_list` are  assumed to have attribute attr (of same type)

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
        # Select adjacency matrix and number of nodes
        # adj_mat_csr = np.copy(self.G_adj_mat_list[G_ind])
        # n_nodes = self.G_n_nodes_list[G_ind]
        
        # Select the graph 
        G_ind = self.G_index_list[idx]
        G = self.G_list[G_ind].copy()
        
        # Compute adjacency matrix and retrieve array of node features 
        # (starting by reordering nodes randomly)
        # - use only random generator from torch -> reproducibility is then
        #   guaranteed by setting `torch.random.manual_seed()` 
        # seq = np.random.permutation(G.number_of_nodes()) # based on numpy
        seq = torch.randperm(G.number_of_nodes()).numpy() 
        adj_mat_csr = networkx.adjacency_matrix(G, seq)
        node_features_array = np.array(list(networkx.get_node_attributes(G, self.attr).values()))[seq]

        if self.node_features_noise_std is not None:
            node_features_array = node_features_array + self.node_features_noise_std * torch.randn(node_features_array.shape).numpy()

        if self.use_bfs:
            G = networkx.from_scipy_sparse_array(adj_mat_csr)
            seq = get_spatial_bfs_sequence(G, node_features_array)
            adj_mat_csr = networkx.adjacency_matrix(G, seq)
            node_features_array = node_features_array[seq]

        # Encode adjacency matrix and node features
        adj_seq_array, node_features_seq_array = encode_adj_and_node_features(adj_mat_csr, node_features_array, max_n_nodes=self.max_n_nodes, max_prev_node=self.max_prev_node)
        # -> adj_seq_array           of shape (n-1, self.max_prev_node), with n <= self.max_n_nodes
        # -> node_features_seq_array of shape (n-1, self.max_prev_node, self.n_node_features)

        # Define x, x_nf and n_nodes (to be delivered as item)
        # x from adj_seq_array
        x = torch.zeros((self.max_n_nodes, self.max_prev_node))#, dtype=torch.float32)
        x[0:adj_seq_array.shape[0], :] = torch.from_numpy(adj_seq_array)
        # x_nf from node_features_seq_array
        x_nf = torch.zeros((self.max_n_nodes, self.max_prev_node, self.n_node_features))#, dtype=torch.float32)
        x_nf[0:node_features_seq_array.shape[0], :, :] = torch.from_numpy(node_features_seq_array)
        # n_nodes: number of nodes taken into account in the encoding
        n_nodes = adj_seq_array.shape[0] + 1

        return x, x_nf, n_nodes

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
# RNN model for graph generation accounting for node features
# -----------------------------------------------------------
# Model similar to the model above, but "differences of node features" are 
# encoded in the data set and added in both RNN_G and RNN_E; if n_node_features 
# is the number of features attached to the graph node, then:
# - RNN_G.input size = max_prev_node * (1+n_node_features)
# - RNN_E.input_size = RNN_E.output_size = 1+n_node_features
#
# See details in the functions below.
# =============================================================================
# ------------------------------------------------------------------------------
def check_rnn_model_graph_gen_with_node_features(rnn_G, rnn_E, data_set):
    """
    Checks a RNN model for graph generation, accounting for node features.

    This function does some checks about the settings of the two imbricated RNN 
    models ((instances of) class :class:`RNN_model`).
    
    Parameters
    ----------
    rnn_G : class :class:`RNN_model`
        network, RNN model at graph level (global state of the graph)
    
    rnn_E : class :class:`RNN_model`
        network, RNN model at edge level (state of connection of one node to 
        the previous ones + differences of node features)
    
    data_set : (instance of) class `Graph_sequence_sampler_data_set_with_node_features`
        data set
    
    Returns
    -------
    ok : bool
        - `ok=True`: if no inconsistency has been found
        - `ok=False`: if an inconsistency has been found
    
    mes : list of str
        list of messages summarizing the issues found (empty list if `ok=True`)
    """
    ok = True
    mes = []

    if rnn_E.hidden_size != rnn_G.output_size:
        ok = False
        mes.append('Issue: rnn_E.hidden_size != rnn_G.output_size')

    if rnn_E.input_size != 1 + data_set.n_node_features:
        ok = False
        mes.append('Issue: rnn_E.input_size != 1 + data_set.n_node_features')

    if rnn_E.output_size != 1 + data_set.n_node_features:
        ok = False
        mes.append('Issue: rnn_E.output_size != 1 + data_set.n_node_features')

    if rnn_G.input_size != data_set.max_prev_node * (1 + data_set.n_node_features):
        ok = False
        mes.append('Issue: rnn_G.input_size != data_set.max_prev_node * (1 + data_set.n_node_features)')
    
    return ok, mes
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def train_rnn_model_graph_gen_with_node_features(
        rnn_G, 
        rnn_E, 
        data_loader,
        optimizer_G,
        optimizer_E,
        lr_scheduler_G=None,
        lr_scheduler_E=None,
        loss_w_edge=1.0,
        loss_w_node_features=1.0,
        random_start_node_features=True,
        return_lr=False,
        return_loss=True,
        num_epochs=10,
        print_epoch=1,
        device=torch.device('cpu')):
    """
    Trains a RNN model for graph generation, accounting for node features.

    The model is constituted of two imbricated RNN models ((instances of) class 
    :class:`RNN_model`)

        - `rnn_G` : RNN model at graph level (global state of the graph): \
        each "time step" x_G[t] consists of \
        rnn_G.input_size = max_prev_node * (1 + n_node_features) values, with
            - x_G[t][i], with i%(1 + n_node_features) = 0: a value 0 or 1 describing \
            the connection (by an edge) from one node to the previous ones
            - x_G[t][i], with i%(1 + n_node_features) in 1, ..., n_node_features: \
            difference of the feature of index i%(1 + n_node_features)-1 between one \
            node and the previous ones
        - `rnn_E` : RNN model at edge level (state of connection of one node to \
        the previous ones + differences of node features): \
        each "time step" x_E[t] consists of rnn_E.input_size = 1 + n_node_features \
        values, with:
            - x_E[t][0]: a value 0 or 1 describing the connection (by an edge) from \
            one node to the previous ones
            - x_E[t][i], with i in 1, ..., n_node_features: \
            difference of the feature of index i-1 between one node and the previous \
            ones

        Moreover, rnn_E.output_size = rnn_E.input_size

    Architecture of the model (RNN_G and RNN_E imbricated):

        - connection with symbol "||" and "==":       show the link from RNN_G to RNN_E
        - connection with symbol "v, ^" and ">>, <<": show the link from RNN_E to RNN_G

    ::
    
                           y_E[0] --+                  y_E[1] --+                 y_E[T_E-1]  >> y_E[*] >>+
                            ^       |                   ^      ...                 ^                      v
                            |       |                   |                          |                      v
                       +---------+  |              +---------+                +---------+                 v
       ++== h_E[0] --> |  RNN_E  |----- h_E[1] --> |  RNN_E  | ----- ...  --> |  RNN_E  | --> h_E[T_E]    v
       ||              +---------+  |              +---------+                +---------+                 v
       ||                   ^       |                   ^                          ^                      v
       ||                   |       |                   |                          |                      v
       ||                  x_E[0]   |                  x_E[1]                     x_E[T_E-1]              v
       ||           =[1,...] (START)|                   ^                          ^                      v
       ||                           |                   |                          |                      v
       ||                           +-------------------+                    ... --+                      v
       ||                                                                                                 v
       ++=========================++      +<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<+
                                  ||      v                                                                
                                 y_G[t]   v                  ...                                           
                                  ^       v                   ^                                            
                                  |       v                   |                                            
                             +---------+  v                +---------+                                     
           ... -- h_G[t] --> |  RNN_G  |----- h_G[t+1] --> |  RNN_G  | --> ...                              
                             +---------+  v                +---------+                                     
                                  ^       v                     ^                                           
                                  |       v                     |                                           
                                 x_G[t]   v                    x_G[t+1]                                     
                                          v                     ^                                           
                                          v                     ^                                           
                                          +>>>>>>>>>>>>>>>>>>>>>+                                           

    Training procedure
        
        - the links feeding the inputs of RNN_E and RNN_G are not used: \
        x_E[t] and x_G[t] at any time step t are provided by the data set
            - for t >= 0: x_G[t] is the sequence of connection (presence of edge) \
            from node t of the graph to the previous nodes: \
            x_G[t][i] = 1 if 0 <= i < min(t, max_prev_node) and an edge exists \
            between node t and node t-(i+1), and x_G[t][i] = 0 otherwise
            - x_G[0] is set to SOS (Start Of Sequence): x_G[0][i] = 1 for all i
            - x_E[0] is set to 1
        - the loss is computed using
            - binary cross-entropy for values determining the presence of edge
            - mse for values corresponding to features differences
            
        between the output y_E of RNN_E and the data provided by the data set (i.e. \
        data in x_G at next time step)

    Parameters
    ----------
    rnn_G : class :class:`RNN_model`
        network, RNN model at graph level (global state of the graph)
    
    rnn_E : class :class:`RNN_model`
        network, RNN model at edge level (state of connection of one node to 
        the previous ones + differences of node features)
    
    data_loader : data loader
        yielding mini-batches (x, x_nf, n_nodes) from a data set, using 
        `batch_first=True`;
        with:

            - B: current batch size
            - N: data_set.max_n_nodes
            - P: data_set.max_prev_node
            - F: data_set.n_node_features

            - x: tensor of shape (B, N, P):
                x[k]: k-th input of the mini-batch (encoded adjacency matrix)
            - x_nf: tensor of shape (B, N, P, F):
                x_nf[k]: k-th input of the mini-batch (encoded node features 
                differences)
            - n_nodes: tensor of shape (B, )
                n_nodes[k]: number of nodes taken into account in the encoding
                `x[k]` (encoded adjacency matrix in the first n_nodes[k]-1 of `x[k]`) 
                and `x_nf[k]` (encoded node features differences in the first 
                n_nodes[k]-1 of `x_nf[k]`) 

    optimizer_G, optimizer_E : optimizer
        updater, torch optimizer for network rnn_G, rnn_E (resp.), e.g. 
        `torch.optim.SGD(net.parameters(), weight_decay=0.001, lr=0.03)`
    
    lr_scheduler_G, lr_scheduler_E : scheduler, optional
        learning rate scheduler for network rnn_G, rnn_E (resp.), e.g. 
        `torch.optim.lr_scheduler.*`
    
    loss_w_edge : float, default 1.0
        weight for loss related to "presence of edge" (binary cross-entropy)
    
    loss_w_node_features : float, default 1.0
        weight for loss related to "features differences" (mse)
    
    random_start_node_features : bool, default: `True`
        - if `True` : start with random node features (standard gaussian)
        - if `False` : start with ones as node features
    
    return_lr : bool, default: `False`
        if `True`: sequence of learning rates for network rnn_G, rnn_E (resp.)
        are returned
    
    return_loss : bool, default: `True`
        if `True`: sequence of losses are returned
    
    num_epochs : int, default: 10
        number of epochs
    
    print_epoch : int, default: 10
        result of every `print_epoch` epoch is displayed in stdout, if 
        `print_epoch > 0`
    
    device : torch device, default: torch.device('cpu')
        device on which the network is trained
    
    Returns
    -------
    loss : list, optional
        returned if `return_loss=True`, training loss of every epoch,
        linear combination of `loss_edge` and `loss_node_features`, 
        list of floats of length `num_epochs`
    
    loss_edge : list, optional
        returned if `return_loss=True`, training loss related to edge
        (presence / absence) of every epoch, 
        list of floats of length `num_epochs`
    
    loss_node_features : list, optional
        returned if `return_loss=True`, training loss related to node
        features (difference btw edge extremities) of every epoch, 
        list of floats of length `num_epochs`
    
    lr_used_G : list, optional
        returned if `return_lr=True`, learning rate for rnn_G used at 
        each epoch, list of floats of length `num_epochs`        
    
    lr_used_E : list, optional
        returned if `return_lr=True`, learning rate for rnn_E used at 
        each epoch, list of floats of length `num_epochs`        
    """
    fname = 'train_rnn_model_graph_gen_with_node_features'

    print('*** Training on', device, '***')

    # Copy models to device
    rnn_G.to(device)
    rnn_E.to(device)

    # Set models in training mode
    rnn_G.train()
    rnn_E.train()

    # Initialize list for loss
    if return_loss:
        loss, loss_edge, loss_node_features = [], [], []

    # Initialize list for lr
    if return_lr:
        lr_used_G = []
        lr_used_E = []

    # Some initializations
    if return_loss:
        loss_len = 0  # reset loss length
        loss_epoch, loss_edge_epoch, loss_node_features_epoch = 0.0, 0.0, 0.0  # reset loss of one epoch
    
    # Train the model
    for epoch in range(num_epochs):
        # rnn_G.train()
        # rnn_E.train()
        # for batch_idx, (x, x_nf, n_nodes) in enumerate(data_loader):
        for x, x_nf, n_nodes in data_loader:
            # Let 
            #   data_set: data set loaded by data_loader
            #   B: current batch size
            #   N: data_set.max_n_nodes
            #   P: data_set.max_prev_node
            #   F: data_set.n_node_features
            # then
            #   x       : tensor of shape (B, N, P)
            #   x_nf    : tensor of shape (B, N, P, F)
            #   n_nodes : tensor of shape (B, ):
            #      number of nodes in the underlying graphs (encoded in x, x_nf), i.e.
            #         x[i, 0:n_nodes[i]-1, :] and x_nf[i, 0:n_nodes[i]-1, :, :]
            #      encoded adjacency matrix and encoded node features differences for the 
            #      i-th element of the mini-batch, and:
            #         x[i, n_nodes[i]-1:, :] = x_nf[i, n_nodes[i]-1:, :, :] = 0 (zeros)
            #
            # Note: rnn_E.input_size = rnn_E.output_size = 1 + F
            
            # Cut x and x_nf tensors along dim 1, wrt. maximum of n_nodes
            n_nodes_max = max(n_nodes)
            x = x[:, 0:n_nodes_max, :]
            x_nf = x_nf[:, 0:n_nodes_max, :, :]

            # rnn_G: input (x_G), output (y_G), packing lengths (pack_lens_G)
            # ---------------------------------------------------------------
            # Initialize y_G
            #    with N' = max(n_nodes)
            #    - add one dimension at the end of x     -> shape (B, N', P, 1)
            #    - concatenate x and x_nf along last dim -> shape (B, N', P, 1+F)
            #    - reshape as (B, N, P*(1+F))
            x_size = x.size() # (B, N', P)
            #n_node_features = x_nf.size(-1) # F
            y_G = torch.cat((x.unsqueeze(-1), x_nf), dim=-1).view(*x_size[:-1], x_size[-1]*rnn_E.input_size)
            # Sort n_nodes in decreasing order (i.e. from the largest to the smallest batch element)
            # -> y_G: tensor containing x and x_nf concateneted, reordered
            n_nodes_G, sort_index = torch.sort(n_nodes, dim=0, descending=True)
            y_G = torch.index_select(y_G, dim=0, index=sort_index).to(device)
            if random_start_node_features:
                # -> x_G: tensor obtained by shifting y_G of one position along the dim 1,
                #         and inserting random numbers (except: ones every (1+F) positions) in the first position (along dim 1)
                tmp = 1.0 + torch.randn((y_G.size(0), 1, y_G.size(2)), device=device)
                # tmp = torch.randn((y_G.size(0), 1, y_G.size(2)), device=device)
                tmp[:, 0, ::rnn_E.input_size] = 1.0
                x_G = torch.cat((tmp, y_G[:,:n_nodes_max-1,:]), dim=1)
            else:
                # -> x_G: tensor obtained by shifting y_G of one position along the dim 1,
                #         and inserting ones in the first position (along dim 1)
                x_G = torch.cat((torch.ones((y_G.size(0), 1, y_G.size(2)), device=device), y_G[:,:n_nodes_max-1,:]), dim=1)
            # -> x_G, y_G of shape (B, N', P*(1+F)), with N' = max(n_nodes)

            # Set length of each element of the mini-batch for rnn_G
            pack_lens_G = n_nodes_G
            
            # rnn_G: hidden varible (state_G)
            # --------------------------------
            state_G = rnn_G.init_state(batch_size=x_G.size(0), device=device) # shape (L, B, H) where L = rnn_G.num_layers, H = rnn_G.hiddensize
                                                                              # (state_G is a 2-tuple if rnn_G is LSTM)

            # rnn_E: input (x_E), output (y_E), packing lengths (pack_lens_E)
            # ---------------------------------------------------------------
            # Pack y_G, wrt to lengths pack_lens_G (already sorted in descending order -> `enforce_sorted=False` below)
            y_G_packed = torch.nn.utils.rnn.pack_padded_sequence(y_G, pack_lens_G, batch_first=True, enforce_sorted=False)
            # -> y_G_packed.data: tensor of shape (sum(pack_lens_G), P)
            # -> y_G_packed.batch_sizes: tensor of dim 1 with sum(y_G_packed.batch_sizes) = sum(pack_lens_G), and:
            #    - first k = y_G_packed.batch_sizes[0] indices (along dim=0) in y_G_packed.data are the 1st entry (index 0) of the k first elements of the mini-batch for rnn_G
            #    - next  k = y_G_packed.batch_sizes[1] indices (along dim=0) in y_G_packed.data are the 2nd entry (index 1) of the k next elements of the mini-batch for rnn_G
            #    - etc.
            # i.e.:
            #    - the k = y_G_packed.batch_sizes[0] first rows in y_G_packed.data correspond to sequences of length 1 (more precisely min(1, P)) [times (1+F) values]
            #    - the k = y_G_packed.batch_sizes[1] next  rows in y_G_packed.data correspond to sequences of length 2 (more precisely min(2, P)) [times (1+F) values]
            #    ...
            
            # Set 
            #   y_E        : y_G_packed.data
            #   pack_lens_E: lengths of sequences in each row of y_E 
            # note: y_G_packed.data on specified device `device`, whereas unique_lens, y_G_packed.batch_sizes on cpu
            y_E = y_G_packed.data # shape (B_E, P*(1+F)) (with B_E = sum(pack_lens_G))
            unique_lens = torch.minimum(torch.arange(1, len(y_G_packed.batch_sizes)+1), torch.full((len(y_G_packed.batch_sizes), ), x_size[-1]))
            #unique_lens = torch.minimum(torch.arange(1, len(y_G_packed.batch_sizes)+1), torch.full((len(y_G_packed.batch_sizes), ), int(y_G_packed.data.size(1)/(1+n_node_features))))
            pack_lens_E = torch.repeat_interleave(unique_lens, y_G_packed.batch_sizes)
            # -> pack_lens_E of length B_E

            # Reverse order of the sequences (y_E along dim=0), so that the sequences corresponding to the rows are of decreasing lengths
            y_E_rev_index = torch.arange(y_E.size(0)-1, -1, -1, device=device)
            y_E = torch.index_select(y_E, dim=0, index=y_E_rev_index)
            pack_lens_E = torch.index_select(pack_lens_E, dim=0, index=y_E_rev_index.to('cpu')) # note: pack_lens_E and index tensor on 'cpu'

            # Reshape last dim of y_E (of size P*(1+F)) into 2 dimensions of size (P, 1+F)
            y_E = y_E.view(y_E.size(0), -1, rnn_E.input_size)
            if random_start_node_features:
                # Set x_E: prepend random numbers (except: one at first position) to y_E along dim 1 (and cut last entry along dim 1)
                tmp = 1.0 + torch.randn((y_E.size(0), 1, rnn_E.input_size), device=device)
                # tmp = torch.randn((y_E.size(0), 1, rnn_E.input_size), device=device)
                tmp[:, 0, 0] = 1.0
                x_E = torch.cat((tmp, y_E[:, :-1, :]), dim=1)
            else:
                # Set x_E: prepend ones to y_E along dim 1 (and cut last entry along dim 1)
                x_E = torch.cat((torch.ones((y_E.size(0), 1, rnn_E.input_size), device=device), y_E[:, :-1, :]), dim=1)
            # -> x_E, y_E of shape (B_E, P, 1+F)
            #    B_E = sum(pack_lens_G) = x_E.size(0) = y_E.size(0): is the current batch size for rnn_E

            # rnn_E: hidden varible (state_E)
            # Set further: output of rnn_G is used as hidden state at layer 0

            # Run rnn_G
            # ---------
            output_G, state_G = rnn_G(x_G, state_G, pack=True, pack_lens=pack_lens_G) # or rnn_G.forward(...)
            # -> output_G of shape (B, N', O_G), with O_G = rnn_G.output_size

            # rnn_E: hidden varible (state_E)
            # -------------------------------
            # Set output of rnn_G as hidden state at layer 0
            #
            # Get packed data of output_G
            output_G_packed_data = torch.nn.utils.rnn.pack_padded_sequence(output_G, pack_lens_G, batch_first=True, enforce_sorted=False).data 
            # -> output_G_packed_data of shape (B_E, O_G), with 
            #    B_E = sum(pack_lens_G) = x_E.size(0) = y_E.size(0)
            #    O_G = rnn_G.output_size = rnn_E.hidden_size = H_E
            #
            # Reverse order (according to what has been done for y_E)
            output_G_packed_data = torch.index_select(output_G_packed_data, dim=0, index=y_E_rev_index)

            state_E = rnn_E.init_state(batch_size=x_E.size(0), device=device) # shape (L_E, B_E, H_E) where L_E = rnn_E.num_layers, H_E = rnn_E.hiddensize
                                                                              # (state_E is a 2-tuple if rnn_E is LSTM)

            if isinstance(state_E, tuple):
                state_E[0][0, ...] = output_G_packed_data
            else:
                state_E[0, ...] = output_G_packed_data

            # # --- Equivalent ---
            # if isinstance(rnn_E.rnn, torch.nn.LSTM):
            #     state_E = (torch.cat((output_G_packed_data.unsqueeze(0), torch.zeros(rnn_E.num_layers-1, output_G_packed_data.size(0), output_G_packed_data.size(1))), dim=0),
            #                torch.zeros(rnn_E.num_layers, output_G_packed_data.size(0), output_G_packed_data.size(1)))
            # else:
            #     state_E = torch.cat((output_G_packed_data.unsqueeze(0), torch.zeros(rnn_E.num_layers-1, output_G_packed_data.size(0), output_G_packed_data.size(1))), dim=0)
            # # ------------------

            # Run rnn_E
            # ---------
            output_E, state_E = rnn_E(x_E, state_E, pack=True, pack_lens=pack_lens_E) # or rnn_E.forward(...)
            # -> output_E of shape (B_E, P, 1+F), with 1+F = O_E = rnn_E.output_size

            # Get packed data of output_E and y_E
            output_E_packed_data = torch.nn.utils.rnn.pack_padded_sequence(output_E, pack_lens_E, batch_first=True).data 
            y_E_packed_data      = torch.nn.utils.rnn.pack_padded_sequence(     y_E, pack_lens_E, batch_first=True).data 
            # -> output_E_packed_data, y_E_packed_data of shape (sum(pack_lens_E), 1+F), with 1+F = O_E = rnn_E.output_size

            # Activate output of rnn_E only on index 0: sigmoid (interpretetion as probabilities)
            y_E_hat_packed_data_0 = torch.nn.functional.sigmoid(output_E_packed_data[:, 0])

            # Compute loss
            # ------------
            # Binary cross-entropy between y_E_hat_packed_data_0 and y_E_packed_data (index 0) is used for "presence of edge"
            loss_edge_mini_batch = torch.nn.functional.binary_cross_entropy(y_E_hat_packed_data_0, y_E_packed_data[:,0])
            # # MSE between output_E_packed_data and y_E_packed_data (from index 1) is used for "node features differences"
            # loss_node_features_mini_batch  = torch.nn.functional.mse_loss(output_E_packed_data[:,1:], y_E_packed_data[:,1:])
            #
            # MSE between output_E_packed_data and y_E_packed_data (from index 1) is used for "node features differences", 
            # only rows where y_E_packed_data[:0]==1 are considered (i.e. when an edge is present)
            ind_edge = y_E_packed_data[:, 0] == 1.0
            loss_node_features_mini_batch = torch.nn.functional.mse_loss(output_E_packed_data[ind_edge, 1:], y_E_packed_data[ind_edge, 1:])
            #
            # Final loss for the minibatch: linear combination of two losses above
            loss_mini_batch = loss_w_edge*loss_edge_mini_batch + loss_w_node_features*loss_node_features_mini_batch

            # Update model
            # ------------
            # Reset gradient in optimized tensors
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            # Compute gradient of loss wrt. to model parameters
            loss_mini_batch.backward()
            # Update for the current mini-batch (one training step)
            optimizer_G.step()
            optimizer_E.step()
            
            if return_loss:
                # Number of entries contributing in loss_mini_batch, i.e. y_E_packed_data.size(0) = sum(pack_lens_E)
                #    note: y_E_packed_data.size(1) = y_E.size(2) = 1
                loss_mini_batch_len = y_E_packed_data.size(0)
                # Update loss for entire epoch (for saving further)
                loss_len += loss_mini_batch_len
                loss_epoch               += float(loss_mini_batch)*loss_mini_batch_len
                loss_edge_epoch          += float(loss_edge_mini_batch)*loss_mini_batch_len
                loss_node_features_epoch += float(loss_node_features_mini_batch)*loss_mini_batch_len

        if return_loss:
            loss.append(loss_epoch/loss_len)
            loss_edge.append(loss_edge_epoch/loss_len)
            loss_node_features.append(loss_node_features_epoch/loss_len)
            loss_len = 0      # reset loss length
            loss_epoch, loss_edge_epoch, loss_node_features_epoch = 0.0, 0.0, 0.0  # reset loss of one epoch

        if return_lr:
            lr_used_G.append(optimizer_G.param_groups[0]['lr'])
            lr_used_E.append(optimizer_E.param_groups[0]['lr'])
        if lr_scheduler_G:
            # lr_used_G.append(lr_scheduler_G.get_last_lr()[0])
            lr_scheduler_G.step()
        if lr_scheduler_E:
            # lr_used_E.append(lr_scheduler_E.get_last_lr()[0])
            lr_scheduler_E.step()

        if print_epoch > 0 and epoch % print_epoch == 0:
            # Print result of current epoch
            s = f'epoch {epoch+1} of {num_epochs}'
            if return_loss:
                s = s + f', loss : {loss[-1]:.6f}, loss_edge : {loss_edge[-1]:.6f}, loss_node_features : {loss_node_features[-1]:.6f}'
            print(s)

    # Set model on cpu
    rnn_G.to(torch.device('cpu'))
    rnn_E.to(torch.device('cpu'))

    out = []
    if return_loss:
        out.append(loss)
        out.append(loss_edge)
        out.append(loss_node_features)
    if return_lr:
        out.append(lr_used_G)
        out.append(lr_used_E)
    out = tuple(out)
    if len(out) == 1:
        out = out[0]
    elif len(out) == 0:
        out = None
    return out
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def generate_graph_with_node_features(
        rnn_G, 
        rnn_E,
        max_n_nodes,
        n_node_features,
        features0=None,
        random_start_node_features=True,
        next_features_mode=None,
        attr='pos',
        n_graph=1,
        force_node1=True,
        return_encoded=False,
        device=torch.device('cpu')):
    """
    Generates one or several graph(s) using a RNN model for graph generation, 
    accounting for node features.

    The model is constituted of two imbricated RNN models ((instances of) class 
    :class:`RNN_model`):

        - `rnn_G` : RNN model at graph level (global state of the graph): \
        each "time step" x_G[t] consists of \
        rnn_G.input_size = max_prev_node * (1 + n_node_features) values, with
            - x_G[t][i], with i%(1 + n_node_features) = 0: a value 0 or 1 describing \
            the connection (by an edge) from one node to the previous ones
            - x_G[t][i], with i%(1 + n_node_features) in 1, ..., n_node_features: \
            difference of the feature of index i%(1 + n_node_features)-1 between one \
            node and the previous ones
        - `rnn_E` : RNN model at edge level (state of connection of one node to \
        the previous ones + differences of node features): \
        each "time step" x_E[t] consists of rnn_E.input_size = 1 + n_node_features \
        values, with:
            - x_E[t][0]: a value 0 or 1 describing the connection (by an edge) from \
            one node to the previous ones
            - x_E[t][i], with i in 1, ..., n_node_features: \
            difference of the feature of index i-1 between one node and the previous \
            ones
            
        Moreover, rnn_E.output_size = rnn_E.input_size

    Architecture of the model (RNN_G and RNN_E imbricated)

        - connection with symbol "||" and "==":       show the link from RNN_G to RNN_E
        - connection with symbol "v, ^" and ">>, <<": show the link from RNN_E to RNN_G

    ::

                           y_E[0] --+                  y_E[1] --+                 y_E[T_E-1]  >> y_E[*] >>+
                            ^       |                   ^      ...                 ^                      v
                            |       |                   |                          |                      v
                       +---------+  |              +---------+                +---------+                 v
       ++== h_E[0] --> |  RNN_E  |----- h_E[1] --> |  RNN_E  | ----- ...  --> |  RNN_E  | --> h_E[T_E]    v
       ||              +---------+  |              +---------+                +---------+                 v
       ||                   ^       |                   ^                          ^                      v
       ||                   |       |                   |                          |                      v
       ||                  x_E[0]   |                  x_E[1]                     x_E[T_E-1]              v
       ||           =[1,...] (START)|                   ^                          ^                      v
       ||                           |                   |                          |                      v
       ||                           +-------------------+                    ... --+                      v
       ||                                                                                                 v
       ++=========================++      +<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<+
                                  ||      v                                                                
                                 y_G[t]   v                  ...                                           
                                  ^       v                   ^                                            
                                  |       v                   |                                            
                             +---------+  v                +---------+                                     
           ... -- h_G[t] --> |  RNN_G  |----- h_G[t+1] --> |  RNN_G  | --> ...                              
                             +---------+  v                +---------+                                     
                                  ^       v                     ^                                           
                                  |       v                     |                                           
                                 x_G[t]   v                    x_G[t+1]                                     
                                          v                     ^                                           
                                          v                     ^                                           
                                          +>>>>>>>>>>>>>>>>>>>>>+                                           

    Generating procedure

        - start with a graph of one node: node 0
        - x_G[0] is set to SOS (Start Of Sequence): x_G[0][i] = 1 for all i
        - then, for any time step t-1 (with t>=1):
            - the outputs y_E[i][0] of RNN_E are interpreted as probabilities \
            (activated by sigmoid)
            - random number (0 or 1) are drawn according to Bernoulli distribution \
            of parameter y_E[i][0] (i.e. value 1 with probability y_E[i][0]), and y_E[i][0] \
            is replaced by the value sampled (0 or 1), before feeding the next step \
            of RNN_E, for each i
            - the vector y_E[*][0] (binary with sampled values) feeds the next step of \
            RNN_G, x_G[t], with, for 0 <= i < min(t, max_prev_node):
                - x_G[t][i*(1+n_node_features)] = y_E[i][0], 
                - x_G[t][i*(1+n_node_features)+j]: y_E[i][j], for j = 1, ..., n_nodes_feature
            - if x_G[t][i*(1+n_node_features)] (i varying) is the vector with only zeros, i.e. \
            EOS (End Of Sequence), the generation of the graph stopped
            - otherwise a new node, node t, is added to the graph with the \
            connection described by x_G[t][i*(1+n_node_features)], and features retrieved \
            using x_G[t][i*(1+n_node_features)+j] (1<=j<=n_nodes_features): this gives a \
            graph of t+1 nodes
        - the procedure stops as soon as EOS is obtained or a maximal \
        number of nodes is reached
    
    Parameters
    ----------
    rnn_G : class :class:`RNN_model`
        network, RNN model at graph level (global state of the graph)
    
    rnn_E : class :class:`RNN_model`
        network, RNN model at edge level (state of connection of one node to 
        the previous ones)
    
    max_n_nodes : int
        maximal number of nodes in the generated graph(s)
    
    n_node_features : int
        number of features attached to nodes (should be equal to 
        `rnn_E.input_size - 1`)
    
    features0 : numpy array of shape (n_node_features, ), optional
        features at node 0;
        by default: `features0 = numpy.zeros(n_node_features)` is used 
    
    random_start_node_features : bool, default: `True`
        - if `True` : start with random node features (standard gaussian)
        - if `False` : start with ones as node features
    
    next_features_mode : str {'sample', 'first', 'mean'}, optional
        mode defining how node features are generated at one node, from the 
        previous connected nodes:

        - 'sample': use a sample node among the previous connected node 
        - 'first': use the first previous connected node 
        - 'mean': use the mean from all the previous connected nodes;
        
        by default (`None`): 'mean' is used
    
    attr : str, default: 'pos'
        name of the attribute attached to nodes, the attribute is a sequence
        of n_node_features floats;
        by default: 'pos' is used (position of the nodes)
    
    n_graph : int, default: 1
        number of graph to be generated
    
    force_node1 : bool, default: `True`
        - if `True` : force creation of node 1 with edge between nodes 0 and 1, \
        i.e. at time step t=0, no sampling (y_E[0] is set to 1 by force)
        - if `False`: no forcing for node 1, i.e. \
        at time step t=0, y_E[0] is sampled
    
    return_encoded : bool, default: `False`
        if `True`: the encoded adjacency matrix is returned
    
    device : torch device, default: torch.device('cpu')
        device on which the network is trained
    
    Returns
    -------
    G_list : list of networkx.Graph object of length `n_graph`
        generated graphs:
        - G_list[k]: k-th graph
    
    adj_seq_array_list : list of 2d numpy arrays of length `n_graph`, optional
        encoded adjacency matrices:

        - adj_seq_array_list[k]: encoded adjacency matrix from which the \
        k-th graph is obtained, 2d numpy array of shape (n-1, max_prev_node) \
        where n is the number of nodes in the graph G_list[k] (with \
        `n <= max_n_nodes`)
    
    node_features_seq_array_list : list of 3d numpy arrays of length `n_graph`, optional
        encoded node features differences:

        - node_features_seq_array_list[k]: encoded node features differences \
        obtained for the k-th graph, 3d numpy array of shape \
        (n-1, max_prev_node, n_node_features) 

    
    Notes
    -----
    If the EOS is obtained for X_G[1], then the generated graph has exactly
    one node, and the encoded adjacency matrix and encoded node features 
    differences are empty (size=0)
    """
    fname = 'generate_graph_with_node_features'

    # Copy models to device
    rnn_G.to(device)
    rnn_E.to(device)

    # Set models in evaluation mode
    rnn_G.eval()
    rnn_E.eval()
    
    # n_node_features = rnn_E.input_size - 1 # could be set automatically
    if features0 is None:
        features0 = np.zeros(n_node_features)

    # Initialization of 
    # - adj_seq          : tensor to store encoded adjacency matrices of generated graphs
    # - node_features_seq: tensor to store encoded node features differences wrt. adj_seq
    max_prev_node = int(rnn_G.input_size/(1+n_node_features))
    adj_seq           = torch.zeros((n_graph, max_n_nodes-1, max_prev_node))
    node_features_seq = torch.zeros((n_graph, max_n_nodes-1, max_prev_node, n_node_features))
    
    # - node_features_arr: tensor to store node features
    #       -> used to adjust node_features_seq at generation of each node
    node_features_arr = torch.zeros((n_graph, max_n_nodes, n_node_features)).to(device)
    node_features_arr[:, 0, :] = torch.from_numpy(features0)

    # rnn_G: input (x_G)
    # ------------------
    # SOS (Start Of Sequence)
    if random_start_node_features:
        x_G = 1.0 + torch.randn((n_graph, 1, rnn_G.input_size), device=device) # shape (B=batch_size=n_graph, N=1, rnn_G.input_size)
        # x_G = torch.randn((n_graph, 1, rnn_G.input_size), device=device) # shape (B=batch_size=n_graph, N=1, rnn_G.input_size)
        x_G[:, 0, ::rnn_E.input_size] = 1.0
    else:
        x_G = torch.ones((n_graph, 1, rnn_G.input_size), device=device) # shape (B=batch_size=n_graph, N=1, rnn_G.input_size)

    # rnn_G: hidden variable (state_G)
    # --------------------------------
    # Initialize
    state_G = rnn_G.init_state(batch_size=x_G.size(0), device=device) # shape (L, B, H) where L = rnn_G.num_layers, H = rnn_G.hiddensize
                                                                      # (state_G is a 2-tuple if rnn_G is LSTM)
    # ####### Warm up (TEST)
    # for i in range(0, 5):#max_n_nodes-1):
    #     # Generate i-th row of encoded adjacency matrices and i-th row of encoded node features differences
    #     # -> adj_seq          [:, i, 0:min(i+1, max_prev_node)]
    #     # -> node_features_seq[:, i, 0:min(i+1, max_prev_node), :]
        
    #     # Run rnn_G
    #     # ---------
    #     with torch.no_grad():
    #         output_G, state_G = rnn_G(x_G, state_G) # or rnn_G.forward(...)
    #         # -> output_G of shape (B, 1, O_G), with O_G = rnn_G.output_size = rnn_E.hidden_size = H_E

    #     # rnn_E: input (x_E)
    #     # ------------------
    #     # SOS (Start Of Sequence)
    #     if random_start_node_features:
    #         x_E = 1.0 + torch.randn((n_graph, 1, rnn_E.input_size), device=device) # shape (B=batch_size=n_graph, N=1, rnn_E.input_size=1+n_node_features)
    #         # x_E = torch.randn((n_graph, 1, rnn_E.input_size), device=device) # shape (B=batch_size=n_graph, N=1, rnn_E.input_size=1+n_node_features)
    #         x_E[:, 0, 0] = 1.0
    #     else:
    #         x_E = torch.ones((n_graph, 1, rnn_E.input_size), device=device) # shape (B=batch_size=n_graph, N=1, rnn_E.input_size=1+n_node_features)

    #     # rnn_E: hidden variable (state_E)
    #     # --------------------------------
    #     # Initialize
    #     state_E = rnn_E.init_state(batch_size=x_E.size(0), device=device) # shape (L_E, B, H_E) where L_E = rnn_E.num_layers, H_E = rnn_E.hiddensize
    #                                                                       # (state_E is a 2-tuple if rnn_E is LSTM)
    #     # Set output of rnn_G as hidden state at layer 0
    #     if isinstance(state_E, tuple):
    #         state_E[0][0, ...] = output_G[:, 0, :]
    #     else:
    #         state_E[0, ...] = output_G[:, 0, :]

    #     x_G = torch.zeros((n_graph, 1, rnn_G.input_size), device=device) # initialization of x_G for next step in rnn_G
    #     for j in range(min(i+1, max_prev_node)):
    #         # Generate absence or presence of edge (value 0 or 1) between node i+1 and node (i+1)-(j+1)
    #         # Run rnn_E
    #         # ---------
    #         with torch.no_grad():
    #             output_E, state_E = rnn_E(x_E, state_E) # or rnn_E.forward(...)
    #             # -> output_E of shape (B, 1, O_E), with O_E = rnn_E.output_size = 1+n_node_features

    #         # Activate output of rnn_E only on index 0: sigmoid (interpretetion as probabilities)
    #         if i == 0 and force_node1:
    #             y_E_hat_0 = torch.ones_like(output_E[:, 0, 0])
    #         else:
    #             y_E_hat_0 = torch.nn.functional.sigmoid(output_E[:, 0, 0])

    #         # Sample according to y_E_hat_0 -> get 0, 1 value in x_E for index 0 for next step in rnn_E
    #         # and set x_E from index 1 to output_E from index 1, where x_E at index 0 is equal to 1
    #         x_E = output_E
    #         x_E[:, 0, 0] = torch.lt(torch.rand(y_E_hat_0.size(), device=device), y_E_hat_0).to(torch.float) # index 0
    #         x_E[:, 0, :] = x_E[:, 0, :] * x_E[:, 0, 0].unsqueeze(-1)

    #         # Update x_G
    #         x_G[:, 0, j*rnn_E.output_size:(j+1)*rnn_E.output_size] = x_E[:, 0, :] 
        
    #     if torch.all(x_G[:, 0, ::rnn_E.output_size] == 0):
    #         print((f'EOS {i}...'))
    #         # EOS is obtained for all graphs
    #         break

    #     # Set i-th row of encoded adjacency matrices and i-th row of encoded node features differences
    #     # adj_seq[:, i, :] = x_G[:, 0, ::rnn_E.output_size]
    #     # node_features_seq[:, i, :, :] = x_G.view(x_G.size(0), 1, int(x_G.size(2)/rnn_E.output_size), rnn_E.output_size)[:, 0, :, 1:]
        
    #     # Set i-th row of encoded adjacency matrices
    #     adj_seq[:, i, :] = x_G[:, 0, ::rnn_E.output_size]
    #     # Set i-th row of encoded node features differences, after adjustment
    #     node_features_diff = x_G.view(x_G.size(0), 1, int(x_G.size(2)/rnn_E.output_size), rnn_E.output_size)[:, 0, :, 1:] 
    #     # -> pointer onto x_G (changes in node_features_diff are effective in x_G)
    #     for n in range(n_graph):
    #         prev_step = torch.where(adj_seq[n, i, :] == 1.0)[0] # index step to previous node that are connected (for graph n)
    #         if len(prev_step) == 0:
    #             continue
    #         node_ind = i - prev_step # index (absolute) of previous connected nodes (for graph n)
    #         node_features_arr[n, i+1, :] = torch.mean(node_features_arr[n, node_ind, :] + node_features_diff[n, prev_step, :], dim=0) # features of node i+1 (new node, for graph n)
    #         node_features_diff[n, prev_step, :] = node_features_arr[n, i+1, :] - node_features_arr[n, node_ind, :] # adjust encoded node features differences (x_G)
    #     node_features_seq[:, i, :, :] = node_features_diff # set i-th row of encoded node features differences

    # # Initialization of 
    # # - adj_seq          : tensor to store encoded adjacency matrices of generated graphs
    # # - node_features_seq: tensor to store encoded node features differences wrt. adj_seq
    # adj_seq           = torch.zeros((n_graph, max_n_nodes-1, max_prev_node))
    # node_features_seq = torch.zeros((n_graph, max_n_nodes-1, max_prev_node, n_node_features))
    
    # # - node_features_arr: tensor to store node features
    # #       -> used to adjust node_features_seq at generation of each node
    # node_features_arr = torch.zeros((n_graph, max_n_nodes, n_node_features)).to(device)
    # node_features_arr[:, 0, :] = torch.from_numpy(features0)

    # # rnn_G: input (x_G)
    # # ------------------
    # # SOS (Start Of Sequence)
    # x_G = torch.ones((n_graph, 1, rnn_G.input_size), device=device) # shape (B=batch_size=n_graph, N=1, rnn_G.input_size)
    # print('Warm-up done')
    # ####### Warm up (TEST)

    for i in range(0, max_n_nodes-1):
        # Generate i-th row of encoded adjacency matrices and i-th row of encoded node features differences
        # -> adj_seq          [:, i, 0:min(i+1, max_prev_node)]
        # -> node_features_seq[:, i, 0:min(i+1, max_prev_node), :]
        
        # Run rnn_G
        # ---------
        with torch.no_grad():
            output_G, state_G = rnn_G(x_G, state_G) # or rnn_G.forward(...)
            # -> output_G of shape (B, 1, O_G), with O_G = rnn_G.output_size = rnn_E.hidden_size = H_E

        # rnn_E: input (x_E)
        # ------------------
        # SOS (Start Of Sequence)
        if random_start_node_features:
            x_E = 1.0 + torch.randn((n_graph, 1, rnn_E.input_size), device=device) # shape (B=batch_size=n_graph, N=1, rnn_E.input_size=1+n_node_features)
            # x_E = torch.randn((n_graph, 1, rnn_E.input_size), device=device) # shape (B=batch_size=n_graph, N=1, rnn_E.input_size=1+n_node_features)
            x_E[:, 0, 0] = 1.0
        else:
            x_E = torch.ones((n_graph, 1, rnn_E.input_size), device=device) # shape (B=batch_size=n_graph, N=1, rnn_E.input_size=1+n_node_features)

        # rnn_E: hidden variable (state_E)
        # --------------------------------
        # Initialize
        state_E = rnn_E.init_state(batch_size=x_E.size(0), device=device) # shape (L_E, B, H_E) where L_E = rnn_E.num_layers, H_E = rnn_E.hiddensize
                                                                          # (state_E is a 2-tuple if rnn_E is LSTM)
        # Set output of rnn_G as hidden state at layer 0
        if isinstance(state_E, tuple):
            state_E[0][0, ...] = output_G[:, 0, :]
        else:
            state_E[0, ...] = output_G[:, 0, :]

        x_G = torch.zeros((n_graph, 1, rnn_G.input_size), device=device) # initialization of x_G for next step in rnn_G
        for j in range(min(i+1, max_prev_node)):
            # Generate absence or presence of edge (value 0 or 1) between node i+1 and node (i+1)-(j+1)
            # Run rnn_E
            # ---------
            with torch.no_grad():
                output_E, state_E = rnn_E(x_E, state_E) # or rnn_E.forward(...)
                # -> output_E of shape (B, 1, O_E), with O_E = rnn_E.output_size = 1+n_node_features

            # Activate output of rnn_E only on index 0: sigmoid (interpretetion as probabilities)
            if i == 0 and force_node1:
                y_E_hat_0 = torch.ones_like(output_E[:, 0, 0])
            else:
                y_E_hat_0 = torch.nn.functional.sigmoid(output_E[:, 0, 0])

            # Sample according to y_E_hat_0 -> get 0, 1 value in x_E for index 0 for next step in rnn_E
            # and set x_E from index 1 to output_E from index 1, where x_E at index 0 is equal to 1
            x_E = output_E
            x_E[:, 0, 0] = torch.lt(torch.rand(y_E_hat_0.size(), device=device), y_E_hat_0).to(torch.float) # index 0
            x_E[:, 0, :] = x_E[:, 0, :] * x_E[:, 0, 0].unsqueeze(-1)

            # Update x_G
            x_G[:, 0, j*rnn_E.output_size:(j+1)*rnn_E.output_size] = x_E[:, 0, :] 
        
        if torch.all(x_G[:, 0, ::rnn_E.output_size] == 0):
            # EOS is obtained for all graphs
            break

        # Set i-th row of encoded adjacency matrices and i-th row of encoded node features differences
        # adj_seq[:, i, :] = x_G[:, 0, ::rnn_E.output_size]
        # node_features_seq[:, i, :, :] = x_G.view(x_G.size(0), 1, int(x_G.size(2)/rnn_E.output_size), rnn_E.output_size)[:, 0, :, 1:]
        
        # Set i-th row of encoded adjacency matrices
        adj_seq[:, i, :] = x_G[:, 0, ::rnn_E.output_size]

        # # Set i-th row of encoded node features differences, after adjustment
        # node_features_diff = x_G.view(x_G.size(0), 1, int(x_G.size(2)/rnn_E.output_size), rnn_E.output_size)[:, 0, :, 1:] 
        # # -> pointer onto x_G (changes in node_features_diff are effective in x_G)
        # for n in range(n_graph):
        #     prev_step = torch.where(adj_seq[n, i, :] == 1.0)[0] # index step to previous node that are connected (for graph n)
        #     if len(prev_step) == 0:
        #         continue
        #     node_ind = i - prev_step # index (absolute) of previous connected nodes (for graph n)
        #     node_features_arr[n, i+1, :] = torch.mean(node_features_arr[n, node_ind, :] + node_features_diff[n, prev_step, :], dim=0) # features of node i+1 (new node, for graph n)
        #     # print('xxx', node_features_diff[n, prev_step, :])
        #     node_features_diff[n, prev_step, :] = node_features_arr[n, i+1, :] - node_features_arr[n, node_ind, :] # adjust encoded node features differences (x_G)
        #     # print('yyy', node_features_diff[n, prev_step, :])
        # node_features_seq[:, i, :, :] = node_features_diff # set i-th row of encoded node features differences

        # Set i-th row of encoded node features differences, after adjustment
        node_features_diff = x_G.view(x_G.size(0), 1, int(x_G.size(2)/rnn_E.output_size), rnn_E.output_size)[:, 0, :, 1:] 
        # -> pointer onto x_G (changes in node_features_diff are effective in x_G)
        if next_features_mode == 'sample':
            for n in range(n_graph):
                prev_step = torch.where(adj_seq[n, i, :] == 1.0)[0] # index step to previous node that are connected (for graph n)
                if len(prev_step) == 0:
                    continue
                node_ind = i - prev_step # index (absolute) of previous connected nodes (for graph n)
                j = np.random.randint(len(prev_step))
                node_features_arr[n, i+1, :] = node_features_arr[n, node_ind[j], :] + node_features_diff[n, prev_step[j], :] # features of node i+1 (new node, for graph n)
                node_features_diff[n, prev_step, :] = node_features_arr[n, i+1, :] - node_features_arr[n, node_ind, :] # adjust encoded node features differences (x_G)
        elif next_features_mode == 'first':
            for n in range(n_graph):
                prev_step = torch.where(adj_seq[n, i, :] == 1.0)[0] # index step to previous node that are connected (for graph n)
                if len(prev_step) == 0:
                    continue
                node_ind = i - prev_step # index (absolute) of previous connected nodes (for graph n)
                j = 0
                node_features_arr[n, i+1, :] = node_features_arr[n, node_ind[j], :] + node_features_diff[n, prev_step[j], :] # features of node i+1 (new node, for graph n)
                node_features_diff[n, prev_step, :] = node_features_arr[n, i+1, :] - node_features_arr[n, node_ind, :] # adjust encoded node features differences (x_G)
        else: # default "mean"
            for n in range(n_graph):
                prev_step = torch.where(adj_seq[n, i, :] == 1.0)[0] # index step to previous node that are connected (for graph n)
                if len(prev_step) == 0:
                    continue
                node_ind = i - prev_step # index (absolute) of previous connected nodes (for graph n)
                node_features_arr[n, i+1, :] = torch.mean(node_features_arr[n, node_ind, :] + node_features_diff[n, prev_step, :], dim=0) # features of node i+1 (new node, for graph n)
                # print('xxx', node_features_diff[n, prev_step, :])
                node_features_diff[n, prev_step, :] = node_features_arr[n, i+1, :] - node_features_arr[n, node_ind, :] # adjust encoded node features differences (x_G)
                # print('yyy', node_features_diff[n, prev_step, :])
        node_features_seq[:, i, :, :] = node_features_diff # set i-th row of encoded node features differences

    # Set model on cpu
    rnn_G.to(torch.device('cpu'))
    rnn_E.to(torch.device('cpu'))

    # Get graphs (and encoded adjacency matrices, relative node position sequence if required)
    G_list = []
    adj_seq_array_list = []
    node_features_seq_array_list = []
    for k in range(n_graph):
        # Get encoded adjacency matrix and relative node position sequence of index k
        adj_seq_array           = adj_seq[k].numpy().astype(int) # shape (max_n_nodes-1, max_prev_node)
        node_features_seq_array = node_features_seq[k].numpy()   # shape (max_n_nodes-1, max_prev_node, n_node_features)
        # Cut it from the first row filled with zeros (if needed)
        ind_all_zeros = np.where(np.all(adj_seq_array==0, axis=1))[0]
        if len(ind_all_zeros):
            adj_seq_array           = adj_seq_array          [0:ind_all_zeros[0], :]
            node_features_seq_array = node_features_seq_array[0:ind_all_zeros[0], :, :]

        if return_encoded:
            adj_seq_array_list.append(adj_seq_array)
            node_features_seq_array_list.append(node_features_seq_array)

        # Decode and build corresponding graph
        adj_mat_csr, node_features_array = decode_adj_and_node_features(adj_seq_array, node_features_seq_array, features0=features0)
        G = networkx.from_scipy_sparse_array(adj_mat_csr)
        node_features_dict = {i: f for i, f in enumerate(node_features_array)}
        networkx.set_node_attributes(G, node_features_dict, attr)
        G_list.append(G)

    out = [G_list]
    if return_encoded:
        out.append(adj_seq_array_list)
        out.append(node_features_seq_array_list)
    out = tuple(out)
    if len(out) == 1:
        out = out[0]
    return out
# ------------------------------------------------------------------------------

# =============================================================================
# RNN + VAE model for graph generation accounting for node features
# -----------------------------------------------------------------
# Model with imbrication of RNN_G <--> RNN_E -> CVAE
#
# See details in the functions below.
# =============================================================================
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
class Variational_encoder_dnf_mlp(torch.nn.Module):
    """
    Variational encoder for difference of node features (MLP model).

    Parameters
    ----------
    in_dim : int
        input dimension of features to be encoded
    
    cond_dim : int
        dimension of conditional vector, zero for simple 
        variational encoder
    
    latent_dim : int
        latent dimension
    
    mlp_hidden_dims: list
        list of dimensions of the MLP hidden layers
    
    batch_norm : bool, default: `False`
        if `True`: Batch normalization is done after each MLP layer
        (except the last layer to get mu and log_var)
    
    dropout : float, default: 0.0
        float in [0, 1), if > 0, probability of Dropout, applied after each 
        MLP layer(+Batch normalization)
        (except the last layer to get mu and log_var)
    
    activation : torch.nn.Module subclass
        non-linearity activation after each MLP layers
        (except the last layer to get mu and log_var)
    """
    def __init__(self, 
                 in_dim,
                 cond_dim, 
                 latent_dim, 
                 mlp_hidden_dims,
                 batch_norm=False, 
                 dropout=0.0, 
                 activation=torch.nn.ReLU()):
        """Constructor method.
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.mlp_hidden_dims = mlp_hidden_dims

        self.mlp_in_dim = in_dim + cond_dim

        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation = activation
        
        seq_dims = [self.mlp_in_dim] + mlp_hidden_dims
        layers = []
        layers.append(torch.nn.Flatten())
        for n_in, n_out in zip(seq_dims[:-1], seq_dims[1:]):
            layers.append(torch.nn.Linear(n_in, n_out))
            layers.append(self.activation)
            if self.batch_norm:
                layers.append(torch.nn.BatchNorm1d(n_out))
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))

        self.layers = torch.nn.Sequential(*layers)

        self.net_mu      = torch.nn.Sequential(torch.nn.Linear(seq_dims[-1], self.latent_dim))
        self.net_log_var = torch.nn.Sequential(torch.nn.Linear(seq_dims[-1], self.latent_dim))

    def forward(self, x, y=None):
        """
        Parameters
        ----------
        x : tensor
            1d tensor (unbatched) or 2d tensor, element(s) to encode
    
        y : tensor, optional
            1d tensor (unbatched) or 2d tensor, conditional vector

        Returns
        -------
        mu : tensor
            2d tensor (even if unbatched input), mean of encoded element(s)
        
        log_var : tensor
            2d tensor (even if unbatched input), log of variance of encoded element(s)
        """
        if y is None:
            xe = self.layers(x)
        else:
            xe = self.layers(torch.cat((x, y), dim=y.ndim-1))
        mu = self.net_mu(xe)
        log_var = self.net_log_var(xe)
        return mu, log_var
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class Decoder_dnf_mlp(torch.nn.Module):
    """
    Decoder for difference of node features (MLP model).

    Parameters
    ----------
    out_dim : int
        output dimension of features (decoded)
    
    cond_dim : int
        dimension of conditional vector, zero for simple decoder
    
    latent_dim : int
        latent dimension
    
    mlp_hidden_dims: list
        list of dimensions of the MLP hidden layers
    
    batch_norm : bool, default: `False`
        if `True`: Batch normalization is done after each MLP layer
        (except the last layer to get decoded features)
    
    dropout : float, default: 0.0
        float in [0, 1), if > 0, probability of Dropout, applied after each 
        MLP layer(+Batch normalization)
        (except the last layer to get decoded features)
    
    activation : torch.nn.Module subclass
        non-linearity activation after each MLP layers
        (except the last layer to get decoded features)
    """
    def __init__(self, 
                 out_dim, 
                 cond_dim, 
                 latent_dim,
                 mlp_hidden_dims,
                 batch_norm=False, 
                 dropout=0.0, 
                 activation=torch.nn.ReLU()):
        """Constructor method.
        """
        super().__init__()

        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.mlp_hidden_dims = mlp_hidden_dims

        self.mlp_in_dim = latent_dim + cond_dim

        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation = activation
        
        seq_dims = [self.mlp_in_dim] + mlp_hidden_dims
        layers = []
        layers.append(torch.nn.Flatten())
        for n_in, n_out in zip(seq_dims[:-1], seq_dims[1:]):
            layers.append(torch.nn.Linear(n_in, n_out))
            layers.append(self.activation)
            if self.batch_norm:
                layers.append(torch.nn.BatchNorm1d(n_out))
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))

        layers.append(torch.nn.Linear(seq_dims[-1], out_dim))
        self.net = torch.nn.Sequential(*layers)
        
    def forward(self, x, y=None):
        """
        Parameters
        ----------
        x : tensor
            1d tensor (unbatched) or 2d tensor, element(s) to decode
        
        y : tensor, optional
            1d tensor (unbatched) or 2d tensor, conditional vector

        Returns
        -------
        output : tensor
            1d tensor (unbatched) or 2d tensor, decoded element
        """
        if y is None:
            output = self.net(x)
        else:
            output = self.net(torch.cat((x, y), dim=y.ndim-1))
        return output
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class CVAE_dnf_mlp(torch.nn.Module):
    """
    Conditional variational autoencoder for difference of node features (MLP model).

    Parameters
    ----------
    dim : int
        dimension of features (encoded / decoded)
    
    cond_dim : int
        dimension of conditional vector, zero for simple decoder
    
    latent_dim : int
        latent dimension
    
    enc_mlp_hidden_dims: list
        list of dimensions of the MLP hidden layers, for encoder
    
    dec_mlp_hidden_dims: list
        list of dimensions of the MLP hidden layers, for decoder
    
    enc_activation : torch.nn.Module subclass
        non-linearity activation, for encoder
    
    dec_activation : torch.nn.Module subclass
        non-linearity activation, for decoder
    """
    def __init__(self, 
                 dim, 
                 cond_dim, 
                 latent_dim,
                 enc_mlp_hidden_dims,
                 dec_mlp_hidden_dims,
                 enc_activation,
                 dec_activation):
        """Constructor method.
        """        
        super().__init__()
        
        self.dim = dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.enc_mlp_hidden_dims = enc_mlp_hidden_dims
        self.dec_mlp_hidden_dims = dec_mlp_hidden_dims
        self.enc_activation = enc_activation
        self.dec_activation = dec_activation

        self.encoder = Variational_encoder_dnf_mlp(self.dim, self.cond_dim, self.latent_dim, self.enc_mlp_hidden_dims, self.enc_activation)
        self.decoder = Decoder_dnf_mlp(self.dim, self.cond_dim, self.latent_dim, self.dec_mlp_hidden_dims, self.dec_activation)
        
    def sample(self, mu, log_var):
        """
        Parameters
        ----------
        mu : tensor (of same size as `log_var`)
            each entry is the mean of a normal distribution
        
        log_var : tensors (of same size)
            each entry is the log of variance of a normal distribution

        Returns
        -------
        z : tensor (of same size as `mu`, `log_var`)
            sample in the normal distribution determined by `mu`, `log_var`
        """
        std = torch.exp(0.5*log_var) # tensor of standard deviation
        eps = torch.randn_like(std)  # tensor ~ N(0, 1)
        return mu + eps*std
        # eps = torch.randn((n, *std.size()), device=std.device) # tensor of n sample in N(0,1_("dim of std"))
        # return mu + torch.mean(eps*std, dim=0) # mean of n samples in N(mu,1_("dim of std"))

    def forward(self, x, y):
        """
        Parameters
        ----------
        x : tensor
            1d tensor (unbatched) or 2d tensor, element(s)
        
        y : tensor
            1d tensor (unbatched) or 2d tensor, conditional vector(s)

        Returns
        -------
        x_hat : tensor
            1d tensor (unbatched) or 2d tensor, element(s) reconstructed
            (input encoded, sampled then decoded)
        
        mu : tensor
            1d tensor (unbatched) or 2d tensor, mean of encoded element(s)
        
        log_var : tensor
            1d tensor (unbatched) or 2d tensor, log of variance of encoded element(s)
        """
        mu, log_var = self.encoder(x, y)
        # if self.training:
        #     z = self.sample(mu, log_var)
        # else:
        #     z = mu
        z = self.sample(mu, log_var)
        x_hat = self.decoder(z, y)
        return x_hat, mu, log_var

    def generate(self, y, device=None):
        """
        Parameters
        ----------
        y : tensor
            1d tensor (unbatched) or 2d tensor, conditional vector(s)
        
        device : torch device, optional
            device on which the model is hosted

        Returns
        -------
        x_hat : tensor
            1d tensor (unbatched) or 2d tensor, generated element(s) (conditional to `y`)
        """
        if y.ndim == 1:
            z = torch.randn(self.latent_dim, device=device)
        else:
            z = torch.randn((y.size(0),self.latent_dim), device=device)
        x_hat = self.decoder(z, y)
        return x_hat
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def train_rnn_model_graph_gen_cvae_with_node_features(
        rnn_G, 
        rnn_E,
        cvae,
        data_loader,
        optimizer_G,
        optimizer_E,
        optimizer_cvae,
        lr_scheduler_G=None,
        lr_scheduler_E=None,
        lr_scheduler_cvae=None,
        loss_w_edge=1.0,
        loss_w_node_features=1.0,
        return_lr=False,
        return_loss=True,
        num_epochs=10,
        print_epoch=1,
        device=torch.device('cpu')):
    """
    Trains a RNN model for graph generation, accounting for node features.

    The model is constituted of two imbricated RNN models ((instances of) class 
    :class:`RNN_model`) + a conditional variational auto-encoder:

        - `rnn_G` : RNN model at graph level (global state of the graph): \
        each "time step" consists of a sequence of 0 or 1 describing the \
        connection (by an edge) from one node to the previous ones
        - `rnn_E` : RNN model at edge level (state of connection of one node to \
        the previous ones): \
        each "time step" consists of a value 0 or 1 describing the connection \
        (by an edge) from one given node to the previous ones
        - `cvae` : conditional VAE that encodes / decodes the node features differences \
        conditioned: 
            - to edge connections given by the output sequence of `rnn_E` and
            - to hidden state of the `rnn_E`

    Architecture of the model (RNN_G and RNN_E imbricated)

        - connection with symbol "||" and "==":       show the link from RNN_G to RNN_E
        - connection with symbol "v, ^" and ">>, <<": show the link from RNN_E to RNN_G

    ::
    
                           y_E[0] --+                  y_E[1] --+                 y_E[T_E-1]  >> y_E[*] >>+
                            ^       |                   ^      ...                 ^                      v
                            |       |                   |                          |                      v
                       +---------+  |              +---------+                +---------+                 v
       ++== h_E[0] --> |  RNN_E  |----- h_E[1] --> |  RNN_E  | ----- ...  --> |  RNN_E  | --> h_E[T_E]    v
       ||              +---------+  |              +---------+                +---------+                 v
       ||                   ^       |                   ^                          ^                      v
       ||                   |       |                   |                          |                      v
       ||                  x_E[0]   |                  x_E[1]                     x_E[T_E-1]              v
       ||               =[1] (START)|                   ^                          ^                      v
       ||                           |                   |                          |                      v
       ||                           +-------------------+                    ... --+                      v
       ||                                                                                                 v
       ++=========================++      +<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<+
                                  ||      v                                                                
                                 y_G[t]   v                  ...                                           
                                  ^       v                   ^                                            
                                  |       v                   |                                            
                             +---------+  v                +---------+                                     
           ... -- h_G[t] --> |  RNN_G  |----- h_G[t+1] --> |  RNN_G  | --> ...                              
                             +---------+  v                +---------+                                     
                                  ^       v                     ^                                           
                                  |       v                     |                                           
                                 x_G[t]   v                    x_G[t+1]                                     
                                          v                     ^                                           
                                          v                     ^                                           
                                          +>>>>>>>>>>>>>>>>>>>>>+                                           

    The conditional VAE is plugged at the end of RNN_E: with 

        - x_F: feature difference (wrt. edge connection from next node to \
        previous ones, given by y_E[*])
    
    ::

              +--------+ +                                                             + +----------+
              |x_F[*]  | | \        +----+                                           / | |hat_x_F[*]|
              |        | |  \    +--| mu |--+                                       /  | |          |
              |        | |   \   |  +----+  |                         +--------+   /   | |          |
              |        | |ENC |--+          |---+-- z = mu + eps*sd --|z       |  | DEC| |          |
              +........+ |   /   |  +----+  |   |                     +........+   \   | +..........+
        cond. |y_E[*]  | |  /    +--| sd |--+   |               cond. |y_E[*]  |    \  |
              |h_E[T_E]| | /        +----+     eps ~ N(0,1)           |h_E[T_E]|     \ |
              +--------+ +                                            +--------+       +
                                              
    Training procedure
        
        - the links feeding the inputs of RNN_E and RNN_G are not used: \
        x_E[t] and x_G[t] at any time step t are provided by the data set:
            - for t >= 0: x_G[t] is the sequence of connection (presence of edge) \
            from node t of the graph to the previous nodes: \
                x_G[t][i] = 1 if 0 <= i < min(t, max_prev_node) and an edge exists \
                between node t and node t-(i+1), and x_G[t][i] = 0 otherwise
            - x_G[0] is set to SOS (Start Of Sequence): x_G[0][i] = 1 for all i
            - x_E[0] is set to 1
        - the loss is computed using
            - binary cross-entropy for values determining the presence of edge
            - loss of vae (reconstruction loss + Kullback-Leibler div) for the \
            features differences
            
    Parameters
    ----------
    rnn_G : class :class:`RNN_model`
        network, RNN model at graph level (global state of the graph)
    
    rnn_E : class :class:`RNN_model`
        network, RNN model at edge level (state of connection of one node to 
        the previous ones)
    
    cvae : network
        conditional variational auto-encoder (for the differences of node 
        features)
    
    data_loader : data loader
        yielding mini-batches (x, x_nf, n_nodes) from a data set, using 
        `batch_first=True`;
        with:

            - B: current batch size
            - N: data_set.max_n_nodes
            - P: data_set.max_prev_node
            - F: data_set.n_node_features

            - x: tensor of shape (B, N, P):
                x[k]: k-th input of the mini-batch (encoded adjacency matrix)
            - x_nf: tensor of shape (B, N, P, F):
                x_nf[k]: k-th input of the mini-batch (encoded node features 
                differences)
            - n_nodes: tensor of shape (B, )
                n_nodes[k]: number of nodes taken into account in the encoding
                `x[k]` (encoded adjacency matrix in the first n_nodes[k]-1 of `x[k]`) 
                and `x_nf[k]` (encoded node features differences in the first 
                n_nodes[k]-1 of `x_nf[k]`) 

    optimizer_G, optimizer_E, optimizer_cvae : optimizer
        updater, torch optimizer for network rnn_G, rnn_E, cvae (resp.), e.g. 
        `torch.optim.SGD(net.parameters(), weight_decay=0.001, lr=0.03)`
    
    lr_scheduler_G, lr_scheduler_E, lr_scheduler_cvae : scheduler, optional
        learning rate scheduler for network rnn_G, rnn_E, cvae (resp.), e.g. 
        `torch.optim.lr_scheduler.*`
    
    loss_w_edge : float, default 1.0
        weight for loss related to "presence of edge" (binary cross-entropy)
    
    loss_w_node_features : float, default 1.0
        weight for loss related to "features differences" (vae: rec. loss + kld)
    
    return_lr : bool, default: `False`
        if `True`: sequence of learning rates for network rnn_G, rnn_E, cvae 
        (resp.) are returned
    
    return_loss : bool, default: `True`
        if `True`: sequence of losses are returned
    
    num_epochs : int, default: 10
        number of epochs
    
    print_epoch : int, default: 10
        result of every `print_epoch` epoch is displayed in stdout, if 
        `print_epoch > 0`
    
    device : torch device, default: torch.device('cpu')
        device on which the network is trained
    
    Returns
    -------
    loss : list, optional
        returned if `return_loss=True`, training loss of every epoch,
        linear combination of `loss_edge` and `loss_node_features`, 
        list of floats of length `num_epochs`
    
    loss_edge : list, optional
        returned if `return_loss=True`, training loss related to edge
        (presence / absence) of every epoch, 
        list of floats of length `num_epochs`
    
    loss_node_features : list, optional
        returned if `return_loss=True`, training loss related to node
        features (loss of vae: rec. loss + kld) of every epoch, 
        list of floats of length `num_epochs`
    
    lr_used_G : list, optional
        returned if `return_lr=True`, learning rate for rnn_G used at 
        each epoch, list of floats of length `num_epochs`        
    
    lr_used_E : list, optional
        returned if `return_lr=True`, learning rate for rnn_E used at 
        each epoch, list of floats of length `num_epochs`        
    
    lr_used_cvae : list, optional
        returned if `return_lr=True`, learning rate for cvae used at 
        each epoch, list of floats of length `num_epochs`        
    """
    fname = 'train_rnn_model_graph_gen_cvae_with_node_features'

    print('*** Training on', device, '***')

    # Copy models to device
    rnn_G.to(device)
    rnn_E.to(device)
    cvae.to(device)

    # Set models in training mode
    rnn_G.train()
    rnn_E.train()
    cvae.train()

    # Initialize list for loss
    if return_loss:
        loss, loss_edge, loss_node_features = [], [], []

    # Initialize list for lr
    if return_lr:
        lr_used_G = []
        lr_used_E = []
        lr_used_cvae = []
    # Some initializations
    if return_loss:
        loss_len = 0  # reset loss length
        loss_epoch, loss_edge_epoch, loss_node_features_epoch = 0.0, 0.0, 0.0  # reset loss of one epoch
    
    # Train the model
    for epoch in range(num_epochs):
        # rnn_G.train()
        # rnn_E.train()
        # for batch_idx, (x, x_nf, n_nodes) in enumerate(data_loader):
        for x, x_nf, n_nodes in data_loader:
            # Let 
            #   data_set: data set loaded by data_loader
            #   B: current batch size
            #   N: data_set.max_n_nodes
            #   P: data_set.max_prev_node
            #   F: data_set.n_node_features
            # then
            #   x       : tensor of shape (B, N, P)
            #   x_nf    : tensor of shape (B, N, P, F)
            #   n_nodes : tensor of shape (B, ):
            #      number of nodes in the underlying graphs (encoded in x, x_nf), i.e.
            #         x[i, 0:n_nodes[i]-1, :] and x_nf[i, 0:n_nodes[i]-1, :, :]
            #      encoded adjacency matrix and encoded node features differences for the 
            #      i-th element of the mini-batch, and:
            #         x[i, n_nodes[i]-1:, :] = x_nf[i, n_nodes[i]-1:, :, :] = 0 (zeros)
            #
            # Note: rnn_E.input_size = rnn_E.output_size = 1
            
            # Cut x tensor along dim 1, wrt. maximum of n_nodes
            n_nodes_max = max(n_nodes)
            x = x[:, 0:n_nodes_max, :]
            x_nf = x_nf[:, 0:n_nodes_max, :, :]
            
            # rnn_G: input (x_G), output (y_G), packing lengths (pack_lens_G)
            # ---------------------------------------------------------------
            # Sort x, n_nodes, such that n_nodes is in decreasing order
            # (i.e. from the largest to the smallest batch element)
            # -> y_G: tensor containing x reordered
            # -> x_G: tensor obtained by shifting y_G of one position along the dim 1,
            #         and inserting ones in the first position (along dim 1)
            n_nodes_G, sort_index = torch.sort(n_nodes, dim=0, descending=True)
            y_G = torch.index_select(x, dim=0, index=sort_index).to(device)
            x_G = torch.cat((torch.ones((y_G.size(0), 1, y_G.size(2)), device=device), y_G[:,:n_nodes_max-1,:]), dim=1)
            # --- or: 
            # x_G = torch.ones_like(y_G, device=device)
            # x_G[:,1:n_nodes_max,:] = y_G[:, :n_nodes_max-1, :]
            # ---            
            # -> x_G, y_G of shape (B, N', P), with N' = max(n_nodes)

            # Set length of each element of the mini-batch for rnn_G
            pack_lens_G = n_nodes_G
            
            # rnn_G: hidden varible (state_G)
            # --------------------------------
            state_G = rnn_G.init_state(batch_size=x_G.size(0), device=device) # shape (L, B, H) where L = rnn_G.num_layers, H = rnn_G.hiddensize
                                                                              # (state_G is a 2-tuple if rnn_G is LSTM)

            # rnn_E: input (x_E), output (y_E), packing lengths (pack_lens_E)
            # ---------------------------------------------------------------
            # Pack y_G, wrt to lengths pack_lens_G (already sorted in descending order -> `enforce_sorted=False` below)
            y_G_packed = torch.nn.utils.rnn.pack_padded_sequence(y_G, pack_lens_G, batch_first=True, enforce_sorted=False)
            # -> y_G_packed.data: tensor of shape (sum(pack_lens_G), P)
            # -> y_G_packed.batch_sizes: tensor of dim 1 with sum(y_G_packed.batch_sizes) = sum(pack_lens_G), and:
            #    - first k = y_G_packed.batch_sizes[0] indices (along dim=0) in y_G_packed.data are the 1st entry (index 0) of the k first elements of the mini-batch for rnn_G
            #    - next  k = y_G_packed.batch_sizes[1] indices (along dim=0) in y_G_packed.data are the 2nd entry (index 1) of the k next elements of the mini-batch for rnn_G
            #    - etc.
            # i.e.:
            #    - the k = y_G_packed.batch_sizes[0] first rows in y_G_packed.data correspond to sequences of length 1 (more precisely min(1, P))
            #    - the k = y_G_packed.batch_sizes[1] next  rows in y_G_packed.data correspond to sequences of length 2 (more precisely min(2, P))
            #    ...
            
            # Set 
            #   y_E        : y_G_packed.data
            #   pack_lens_E: lengths of sequences in each row of y_E 
            # note: y_G_packed.data on specified device `device`, whereas unique_lens, y_G_packed.batch_sizes on cpu
            y_E = y_G_packed.data # shape (B_E, P) (with B_E = sum(pack_lens_G))
            unique_lens = torch.minimum(torch.arange(1, len(y_G_packed.batch_sizes)+1), torch.full((len(y_G_packed.batch_sizes), ), y_G_packed.data.size(1)))
            pack_lens_E = torch.repeat_interleave(unique_lens, y_G_packed.batch_sizes)
            # -> pack_lens_E of length B_E
            
            # Reverse order of the sequences (y_E along dim=0), so that the sequences corresponding to the rows are of decreasing lengths
            y_E_rev_index = torch.arange(y_E.size(0)-1, -1, -1, device=device)
            y_E = torch.index_select(y_E, dim=0, index=y_E_rev_index)
            pack_lens_E = torch.index_select(pack_lens_E, dim=0, index=y_E_rev_index.to('cpu')) # note: pack_lens_E and index tensor on 'cpu'

            # Add a dimension (of size 1) at the end to y_E
            # and set x_E: prepend ones to y_E along dim 1 (and cut last entry along dim 1)
            y_E = y_E.unsqueeze(-1)
            x_E = torch.cat((torch.ones((y_E.size(0), 1, 1), device=device), y_E[:, :-1, :]), dim=1)
            # -> x_E, y_E of shape (B_E, P, 1)
            #    B_E = sum(pack_lens_G) = x_E.size(0) = y_E.size(0): is the current batch size for rnn_E
            
            # rnn_E: hidden varible (state_E)
            # Set further: output of rnn_G is used as hidden state at layer 0

            # Run rnn_G
            # ---------
            output_G, state_G = rnn_G(x_G, state_G, pack=True, pack_lens=pack_lens_G) # or rnn_G.forward(...)
            # -> output_G of shape (B, N', O_G), with O_G = rnn_G.output_size

            # rnn_E: hidden varible (state_E)
            # -------------------------------
            # Set output of rnn_G as hidden state at layer 0
            #
            # Get packed data of output_G
            output_G_packed_data = torch.nn.utils.rnn.pack_padded_sequence(output_G, pack_lens_G, batch_first=True, enforce_sorted=False).data 
            # -> output_G_packed_data of shape (B_E, O_G), with 
            #    B_E = sum(pack_lens_G) = x_E.size(0) = y_E.size(0)
            #    O_G = rnn_G.output_size = rnn_E.hidden_size = H_E
            #
            # Reverse order (according to what has been done for y_E)
            output_G_packed_data = torch.index_select(output_G_packed_data, dim=0, index=y_E_rev_index)

            state_E = rnn_E.init_state(batch_size=x_E.size(0), device=device) # shape (L_E, B_E, H_E) where L_E = rnn_E.num_layers, H_E = rnn_E.hiddensize
                                                                              # (state_E is a 2-tuple if rnn_E is LSTM)

            if isinstance(state_E, tuple):
                state_E[0][0, ...] = output_G_packed_data
            else:
                state_E[0, ...] = output_G_packed_data

            # # --- Equivalent ---
            # if isinstance(rnn_E.rnn, torch.nn.LSTM):
            #     state_E = (torch.cat((output_G_packed_data.unsqueeze(0), torch.zeros(rnn_E.num_layers-1, output_G_packed_data.size(0), output_G_packed_data.size(1))), dim=0),
            #                torch.zeros(rnn_E.num_layers, output_G_packed_data.size(0), output_G_packed_data.size(1)))
            # else:
            #     state_E = torch.cat((output_G_packed_data.unsqueeze(0), torch.zeros(rnn_E.num_layers-1, output_G_packed_data.size(0), output_G_packed_data.size(1))), dim=0)
            # # ------------------

            # Run rnn_E
            # ---------
            output_E, state_E = rnn_E(x_E, state_E, pack=True, pack_lens=pack_lens_E) # or rnn_E.forward(...)
            # -> output_E of shape (B_E, P, 1), with 1 = O_E = rnn_E.output_size=1

            # Get packed data of output_E and y_E
            output_E_packed_data = torch.nn.utils.rnn.pack_padded_sequence(output_E, pack_lens_E, batch_first=True).data 
            y_E_packed_data      = torch.nn.utils.rnn.pack_padded_sequence(     y_E, pack_lens_E, batch_first=True).data 
            # -> output_E_packed_data, y_E_packed_data of shape (sum(pack_lens_E), 1), with 1 = O_E = rnn_E.output_size=1

            # Activate output of rnn_E: sigmoid (interpretetion as probabilities)
            y_E_hat_packed_data = torch.nn.functional.sigmoid(output_E_packed_data)

            # Compute loss (for topology of the graph)
            # ------------
            # Binary cross-entropy between y_E_hat_packed_data and y_E_packed_data is used
            loss_edge_mini_batch = torch.nn.functional.binary_cross_entropy(y_E_hat_packed_data, y_E_packed_data)

            # CVAE on node features diff
            # --------------------------
            # y_E : shape (B_E, P, 1), set from `x` (encoded adjacency matrix)
            #       contains B_E batches of shape (P, 1) corresponding to presence or absence of connection (0 / 1) to previous nodes
            # Apply operations to `x_nf` (encoded node features differences) to get:
            # x_F : shape (B_E, P, F), where x_F[i, j, : ] are the features differences corresponding to y_E[i, j]
            x_F = torch.index_select(x_nf, dim=0, index=sort_index).to(device) # shape (B, N', P, F)
            x_F = torch.nn.utils.rnn.pack_padded_sequence(x_F, pack_lens_G, batch_first=True, enforce_sorted=False).data # shape (B_E, P, F)
            x_F = torch.index_select(x_F, dim=0, index=y_E_rev_index) # reorder as done for y_E
            x_F_n = x_F.size(-1) # save number of features (F)
            # Reshape x_F : new shape (B_E, P*F)
            x_F = x_F.view(x_F.size(0), -1) # shape (B_E, P*F)
            # Set y_F the conditional entries for CVAE
            if isinstance(state_E, tuple):
                y_F = state_E[0][-1]
            else:
                y_F = state_E[-1]
            # -> y_F : last layer of hidden state of rnn_E, shape (B_E, H_E)
            y_F = torch.cat((y_F, y_E.view(y_E.size(0), -1)), dim=1)
            # -> y_F : shape (B_E, H_E + P)
            # Run the CVAE
            x_F_hat, mu, log_var = cvae(x_F, y_F)

            # Compute loss of the CVAE
            # ------------------------
            # # MSE between x_F_hat and x_F is used for "reconstruction loss"
            # rec_loss = torch.nn.MSELoss(reduction='none')(x_F_hat, x_F).sum(dim=1).mean() # reconstruction loss
            #
            # # MSE between x_F_hat and x_F is used for "reconstruction loss"
            # # only entries corresponding to y_E==1 are considered (i.e. when an edge is present)
            ind_edge = torch.repeat_interleave(y_E.squeeze(-1), x_F_n, dim=1) # tensor of bool of size (B_E, P*F) (as x_F, x_F_hat)
            rec_loss = torch.nn.MSELoss(reduction='none')(x_F_hat*ind_edge, x_F*ind_edge).sum(dim=1).mean() # reconstruction loss

            kld = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mu**2, dim=1).mean() # kullback-leibler divergence with N(0,1)
            loss_node_features_mini_batch = rec_loss + kld

            # Final loss
            # ----------
            loss_mini_batch = loss_w_edge*loss_edge_mini_batch + loss_w_node_features*loss_node_features_mini_batch

            # Update model
            # ------------
            # Reset gradient in optimized tensors
            optimizer_G.zero_grad()
            optimizer_E.zero_grad()
            optimizer_cvae.zero_grad()
            # Compute gradient of loss wrt. to model parameters
            loss_mini_batch.backward()
            # Update for the current mini-batch (one training step)
            optimizer_G.step()
            optimizer_E.step()
            optimizer_cvae.step()
            
            if return_loss:
                # Number of entries contributing in loss_mini_batch, i.e. y_E_packed_data.size(0) = sum(pack_lens_E)
                #    note: y_E_packed_data.size(1) = y_E.size(2) = 1
                loss_mini_batch_len = y_E_packed_data.size(0)
                # Update loss for entire epoch (for saving further)
                loss_len += loss_mini_batch_len
                loss_epoch               += float(loss_mini_batch)*loss_mini_batch_len
                loss_edge_epoch          += float(loss_edge_mini_batch)*loss_mini_batch_len
                loss_node_features_epoch += float(loss_node_features_mini_batch)*loss_mini_batch_len

        if return_loss:
            loss.append(loss_epoch/loss_len)
            loss_edge.append(loss_edge_epoch/loss_len)
            loss_node_features.append(loss_node_features_epoch/loss_len)
            loss_len = 0      # reset loss length
            loss_epoch, loss_edge_epoch, loss_node_features_epoch = 0.0, 0.0, 0.0  # reset loss of one epoch

        if return_lr:
            lr_used_G.append(optimizer_G.param_groups[0]['lr'])
            lr_used_E.append(optimizer_E.param_groups[0]['lr'])
            lr_used_cvae.append(optimizer_cvae.param_groups[0]['lr'])
        if lr_scheduler_G:
            # lr_used_G.append(lr_scheduler_G.get_last_lr()[0])
            lr_scheduler_G.step()
        if lr_scheduler_E:
            # lr_used_E.append(lr_scheduler_E.get_last_lr()[0])
            lr_scheduler_E.step()
        if lr_scheduler_cvae:
            # lr_used_cvae.append(lr_scheduler_E.get_last_lr()[0])
            lr_scheduler_cvae.step()

        if print_epoch > 0 and epoch % print_epoch == 0:
            # Print result of current epoch
            s = f'epoch {epoch+1} of {num_epochs}'
            if return_loss:
                s = s + f', loss : {loss[-1]:.6f}, loss_edge : {loss_edge[-1]:.6f}, loss_node_features : {loss_node_features[-1]:.6f}'
            print(s)

    # Set model on cpu
    rnn_G.to(torch.device('cpu'))
    rnn_E.to(torch.device('cpu'))
    cvae.to(torch.device('cpu'))

    out = []
    if return_loss:
        out.append(loss)
        out.append(loss_edge)
        out.append(loss_node_features)
    if return_lr:
        out.append(lr_used_G)
        out.append(lr_used_E)
        out.append(lr_used_cvae)
    out = tuple(out)
    if len(out) == 1:
        out = out[0]
    elif len(out) == 0:
        out = None
    return out
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def generate_graph_cvae_with_node_features(
        rnn_G, 
        rnn_E,
        cvae,
        max_n_nodes,
        n_node_features,
        features0=None,
        next_features_mode=None,
        attr='pos',
        n_graph=1,
        return_encoded=False,
        device=torch.device('cpu')):
    """
    Generates one or several graph(s) using a RNN model for graph generation, 
    accounting for node features.

    The model is constituted of two imbricated RNN models ((instances of) class 
    :class:`RNN_model`) + a conditional variational auto-encoder:

        - `rnn_G` : RNN model at graph level (global state of the graph): \
        each "time step" consists of a sequence of 0 or 1 describing the \
        connection (by an edge) from one node to the previous ones
        - `rnn_E` : RNN model at edge level (state of connection of one node to \
        the previous ones): \
        each "time step" consists of a value 0 or 1 describing the connection \
        (by an edge) from one given node to the previous ones
        - `cvae` : conditional VAE that encodes / decodes the node features differences \
        conditioned: 
            - to edge connections given by the output sequence of `rnn_E` and
            - to hidden state of the `rnn_E`

    Architecture of the model (RNN_G and RNN_E imbricated)

        - connection with symbol "||" and "==":       show the link from RNN_G to RNN_E
        - connection with symbol "v, ^" and ">>, <<": show the link from RNN_E to RNN_G

    ::
    
                           y_E[0] --+                  y_E[1] --+                 y_E[T_E-1]  >> y_E[*] >>+
                            ^       |                   ^      ...                 ^                      v
                            |       |                   |                          |                      v
                       +---------+  |              +---------+                +---------+                 v
       ++== h_E[0] --> |  RNN_E  |----- h_E[1] --> |  RNN_E  | ----- ...  --> |  RNN_E  | --> h_E[T_E]    v
       ||              +---------+  |              +---------+                +---------+                 v
       ||                   ^       |                   ^                          ^                      v
       ||                   |       |                   |                          |                      v
       ||                  x_E[0]   |                  x_E[1]                     x_E[T_E-1]              v
       ||               =[1] (START)|                   ^                          ^                      v
       ||                           |                   |                          |                      v
       ||                           +-------------------+                    ... --+                      v
       ||                                                                                                 v
       ++=========================++      +<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<+
                                  ||      v                                                                
                                 y_G[t]   v                  ...                                           
                                  ^       v                   ^                                            
                                  |       v                   |                                            
                             +---------+  v                +---------+                                     
           ... -- h_G[t] --> |  RNN_G  |----- h_G[t+1] --> |  RNN_G  | --> ...                              
                             +---------+  v                +---------+                                     
                                  ^       v                     ^                                           
                                  |       v                     |                                           
                                 x_G[t]   v                    x_G[t+1]                                     
                                          v                     ^                                           
                                          v                     ^                                           
                                          +>>>>>>>>>>>>>>>>>>>>>+                                           

    The conditional VAE is plugged at the end of RNN_E: with 

        - x_F: feature difference (wrt. edge connection from next node to \
        previous ones, given by y_E[*])
    
    ::

              +--------+ +                                                             + +----------+
              |x_F[*]  | | \        +----+                                           / | |hat_x_F[*]|
              |        | |  \    +--| mu |--+                                       /  | |          |
              |        | |   \   |  +----+  |                         +--------+   /   | |          |
              |        | |ENC |--+          |---+-- z = mu + eps*sd --|z       |  | DEC| |          |
              +........+ |   /   |  +----+  |   |                     +........+   \   | +..........+
        cond. |y_E[*]  | |  /    +--| sd |--+   |               cond. |y_E[*]  |    \  |
              |h_E[T_E]| | /        +----+     eps ~ N(0,1)           |h_E[T_E]|     \ |
              +--------+ +                                            +--------+       +
                                              
    Generating procedure

        - start with a graph of one node: node 0
        - x_G[0] is set to SOS (Start Of Sequence): x_G[0][i] = 1 for all i
        - then, for any time step t-1 (with t>=1)
            - the outputs y_E[i] of RNN_E are interpreted as probabilities \
            (activated by sigmoid)
            - random number (0 or 1) are drawn according to Bernoulli distribution \
            of parameter y_E[i] (i.e. value 1 with probability y_E[i]), and y_E[i] \
            is replaced by the value sampled (0 or 1), before feeding the next step \
            of RNN_E, for each i
            - the vector y_E[*] (binary with sampled values) with the hiddent state \
            of RNN_E (last layer) are concatenated to form the conditional entries \
            for the cvae, which is used in decoder part with sample z to generate \
            feature differences x_F[t] wrt. y_E for the next node (these node features \
            differences are adjusted to be consistent, accounting for the features \
            at the previous nodes)
            - the vector y_E[*] feeds the next step of RNN_G, x_G[t], with
                - x_G[t][i] = y_E[i], for 0 <= i < min(t, max_prev_node), and 
                - x_G[t][i] = 0, for t <= i < max_prev_node (if any)
            - if x_G[t] is the vector with only zeros, i.e. EOS (End Of Sequence), \
            the generation of the graph stopped
            - otherwise a new node, node t, is added to the graph with the \
            connection described by x_G[t] and the features derived from x_F[t]: \
            this gives a graph of t+1 nodes
        - the procedure stops as soon as EOS is obtained or a maximal \
        number of nodes is reached

    Parameters
    ----------
    rnn_G : class :class:`RNN_model`
        network, RNN model at graph level (global state of the graph)
    
    rnn_E : class :class:`RNN_model`
        network, RNN model at edge level (state of connection of one node to 
        the previous ones)
    
    cvae : network
        conditional variational auto-encoder (for the differences of node 
        features)
    
    max_n_nodes : int
        maximal number of nodes in the generated graph(s)
    
    n_node_features : int
        number of features attached to nodes (should be equal to 
        `rnn_E.input_size - 1`)
    
    features0 : numpy array of shape (n_node_features, ), optional
        features at node 0;
        by default: `features0 = numpy.zeros(n_node_features)` is used 
    
    next_features_mode : str {'sample', 'first', 'mean'}, optional
        mode defining how node features are generated at one node, from the 
        previous connected nodes:

        - 'sample': use a sample node among the previous connected node 
        - 'first': use the first previous connected node 
        - 'mean': use the mean from all the previous connected nodes;
        
        by default (`None`): 'mean' is used
    
    attr : str, default: 'pos'
        name of the attribute attached to nodes, the attribute is a sequence
        of n_node_features floats;
        by default: 'pos' is used (position of the nodes)
    
    n_graph : int, default: 1
        number of graph to be generated
    
    return_encoded : bool, default: `False`
        if `True`: the encoded adjacency matrix is returned
    
    device : torch device, default: torch.device('cpu')
        device on which the network is trained
    
    Returns
    -------
    G_list : list of networkx.Graph object of length `n_graph`
        generated graphs:

        - G_list[k]: k-th graph
    
    adj_seq_array_list : list of 2d numpy arrays of length `n_graph`, optional
        encoded adjacency matrices:

        - adj_seq_array_list[k]: encoded adjacency matrix from which the \
        k-th graph is obtained, 2d numpy array of shape (n-1, max_prev_node) \
        where n is the number of nodes in the graph G_list[k] (with \
        `n <= max_n_nodes`)

    node_features_seq_array_list : list of 3d numpy arrays of length `n_graph`, optional
        encoded node features differences:
        
        - node_features_seq_array_list[k]: encoded node features differences \
        obtained for the k-th graph, 3d numpy array of shape \
        (n-1, max_prev_node, n_node_features) 

    Notes
    -----
    If the EOS is obtained for X_G[1], then the generated graph has exactly
    one node, and the encoded adjacency matrix and encoded node features 
    differences are empty (size=0)
    """
    fname = 'generate_graph_cvae_with_node_features'

    # Copy models to device
    rnn_G.to(device)
    rnn_E.to(device)
    cvae.to(device)

    # Set models in evaluation mode
    rnn_G.eval()
    rnn_E.eval()
    cvae.eval()
    
    # n_node_features = rnn_E.input_size - 1 # could be set automatically
    if features0 is None:
        features0 = np.zeros(n_node_features)

    # Initialization of 
    # - adj_seq          : tensor to store encoded adjacency matrices of generated graphs
    # - node_features_seq: tensor to store encoded node features differences wrt. adj_seq
    max_prev_node = rnn_G.input_size
    adj_seq           = torch.zeros((n_graph, max_n_nodes-1, max_prev_node))
    node_features_seq = torch.zeros((n_graph, max_n_nodes-1, max_prev_node, n_node_features))
    
    # - node_features_arr: tensor to store node features
    #       -> used to adjust node_features_seq at generation of each node
    node_features_arr = torch.zeros((n_graph, max_n_nodes, n_node_features)).to(device)
    node_features_arr[:, 0, :] = torch.from_numpy(features0)

    # rnn_G: input (x_G)
    # ------------------
    # SOS (Start Of Sequence)
    x_G = torch.ones((n_graph, 1, max_prev_node), device=device) # shape (B=batch_size=n_graph, N=1, rnn_G.input_size=max_prev_node)

    # rnn_G: hidden variable (state_G)
    # --------------------------------
    # Initialize
    state_G = rnn_G.init_state(batch_size=x_G.size(0), device=device) # shape (L, B, H) where L = rnn_G.num_layers, H = rnn_G.hiddensize
                                                                      # (state_G is a 2-tuple if rnn_G is LSTM)
    for i in range(0, max_n_nodes-1):
        # Generate i-th row of encoded adjacency matrices
        # -> adj_seq[:, i, 0:min(i+1, max_prev_node)]
        
        # Run rnn_G
        # ---------
        with torch.no_grad():
            output_G, state_G = rnn_G(x_G, state_G) # or rnn_G.forward(...)
            # -> output_G of shape (B, 1, O_G), with O_G = rnn_G.output_size = rnn_E.hidden_size = H_E

        # rnn_E: input (x_E)
        # ------------------
        # SOS (Start Of Sequence)
        x_E = torch.ones((n_graph, 1, 1), device=device) # shape (B=batch_size=n_graph, N=1, rnn_E.input_size=1)

        # rnn_E: hidden variable (state_E)
        # --------------------------------
        # Initialize
        state_E = rnn_E.init_state(batch_size=x_E.size(0), device=device) # shape (L_E, B, H_E) where L_E = rnn_E.num_layers, H_E = rnn_E.hiddensize
                                                                          # (state_E is a 2-tuple if rnn_E is LSTM)
        # Set output of rnn_G as hidden state at layer 0
        if isinstance(state_E, tuple):
            state_E[0][0, ...] = output_G[:, 0, :]
        else:
            state_E[0, ...] = output_G[:, 0, :]

        x_G = torch.zeros((n_graph, 1, rnn_G.input_size), device=device) # initialization of x_G for next step in rnn_G
        for j in range(min(i+1, max_prev_node)):
            # Generate absence or presence of edge (value 0 or 1) between node i+1 and node (i+1)-(j+1)
            # Run rnn_E
            # ---------
            with torch.no_grad():
                output_E, state_E = rnn_E(x_E, state_E) # or rnn_E.forward(...)
                # -> output_E of shape (B, 1, O_E), with O_E = rnn_E.output_size = 1

            # Activate output of rnn_E: sigmoid (interpretetion as probabilities)
            y_E_hat = torch.nn.functional.sigmoid(output_E)

            # Sample according to y_E_hat -> get 0, 1 value in x_E for next step in rnn_E
            x_E = torch.lt(torch.rand(y_E_hat.size(), device=device), y_E_hat).to(torch.float)

            # Update x_G
            x_G[:, 0, j] = x_E[:, 0, 0] 
        
        if torch.all(x_G[:, 0, :] == 0):
            # EOS is obtained for all graphs
            break

        # Set i-th row of encoded adjacency matrices
        adj_seq[:, i, :] = x_G[:, 0, :]
        # # or : 
        # adj_seq[:, i, 0:min(i+1, max_prev_node)] = x_G[:, 0, 0:min(i+1, max_prev_node)]

        if isinstance(state_E, tuple):
            y_F = state_E[0][-1]
        else:
            y_F = state_E[-1]
        # -> y_F : last layer of hidden state of rnn_E, shape (B, H_E)
        y_F = torch.cat((y_F, x_G[:, 0, :]), dim=1)
        # -> y_F : shape (B_E, H + P)
        with torch.no_grad():
            x_F = cvae.generate(y_F, device=device)
        x_F = x_F.view(node_features_seq.size(0), node_features_seq.size(2), node_features_seq.size(3))

        # Set i-th row of encoded node features differences, after adjustment
        if next_features_mode == 'sample':
            for n in range(n_graph):
                prev_step = torch.where(adj_seq[n, i, :] == 1.0)[0] # index step to previous node that are connected (for graph n)
                if len(prev_step) == 0:
                    continue
                node_ind = i - prev_step # index (absolute) of previous connected nodes (for graph n)
                j = np.random.randint(len(prev_step))
                node_features_arr[n, i+1, :] = node_features_arr[n, node_ind[j], :] + x_F[n, prev_step[j], :] # features of node i+1 (new node, for graph n)
                node_features_seq[n, i, prev_step, :] = (node_features_arr[n, i+1, :] - node_features_arr[n, node_ind, :]).to('cpu') # adjust encoded node features differences (x_G)
        elif next_features_mode == 'first':
            for n in range(n_graph):
                prev_step = torch.where(adj_seq[n, i, :] == 1.0)[0] # index step to previous node that are connected (for graph n)
                if len(prev_step) == 0:
                    continue
                node_ind = i - prev_step # index (absolute) of previous connected nodes (for graph n)
                j = 0
                node_features_arr[n, i+1, :] = node_features_arr[n, node_ind[j], :] + x_F[n, prev_step[j], :] # features of node i+1 (new node, for graph n)
                node_features_seq[n, i, prev_step, :] = (node_features_arr[n, i+1, :] - node_features_arr[n, node_ind, :]).to('cpu') # adjust encoded node features differences (x_G)
        else: # default "mean"
            for n in range(n_graph):
                prev_step = torch.where(adj_seq[n, i, :] == 1.0)[0] # index step to previous node that are connected (for graph n)
                if len(prev_step) == 0:
                    continue
                node_ind = i - prev_step # index (absolute) of previous connected nodes (for graph n)
                node_features_arr[n, i+1, :] = torch.mean(node_features_arr[n, node_ind, :] + x_F[n, prev_step, :], dim=0) # features of node i+1 (new node, for graph n)
                node_features_seq[n, i, prev_step, :] = (node_features_arr[n, i+1, :] - node_features_arr[n, node_ind, :]).to('cpu') # adjust encoded node features differences (x_G)
                # print('xxx', x_F[n, prev_step, :])
                # print('yyy', node_features_seq[n, i, prev_step, :])
        # # Set i-th row of encoded node features differences, after adjustment
        # for n in range(n_graph):
        #     prev_step = torch.where(adj_seq[n, i, :] == 1.0)[0] # index step to previous node that are connected (for graph n)
        #     if len(prev_step) == 0:
        #         continue
        #     node_ind = i - prev_step # index (absolute) of previous connected nodes (for graph n)
        #     node_features_arr[n, i+1, :] = torch.mean(node_features_arr[n, node_ind, :] + x_F[n, prev_step, :], dim=0) # features of node i+1 (new node, for graph n)
        #     node_features_seq[n, i, prev_step, :] = (node_features_arr[n, i+1, :] - node_features_arr[n, node_ind, :]).to('cpu') # adjust encoded node features differences (x_G)
        #     # print('xxx', x_F[n, prev_step, :])
        #     # print('yyy', node_features_seq[n, i, prev_step, :])

    # Set model on cpu
    rnn_G.to(torch.device('cpu'))
    rnn_E.to(torch.device('cpu'))
    cvae.to(torch.device('cpu'))

    # Get graphs (and encoded adjacency matrices, relative node position sequence if required)
    G_list = []
    adj_seq_array_list = []
    node_features_seq_array_list = []
    for k in range(n_graph):
        # Get encoded adjacency matrix and relative node position sequence of index k
        adj_seq_array           = adj_seq[k].numpy().astype(int) # shape (max_n_nodes-1, max_prev_node)
        node_features_seq_array = node_features_seq[k].numpy()   # shape (max_n_nodes-1, max_prev_node, n_node_features)
        # Cut it from the first row filled with zeros (if needed)
        ind_all_zeros = np.where(np.all(adj_seq_array==0, axis=1))[0]
        if len(ind_all_zeros):
            adj_seq_array           = adj_seq_array          [0:ind_all_zeros[0], :]
            node_features_seq_array = node_features_seq_array[0:ind_all_zeros[0], :, :]

        if return_encoded:
            adj_seq_array_list.append(adj_seq_array)
            node_features_seq_array_list.append(node_features_seq_array)

        # Decode and build corresponding graph
        adj_mat_csr, node_features_array = decode_adj_and_node_features(adj_seq_array, node_features_seq_array, features0=features0)
        G = networkx.from_scipy_sparse_array(adj_mat_csr)
        node_features_dict = {i: f for i, f in enumerate(node_features_array)}
        networkx.set_node_attributes(G, node_features_dict, attr)
        G_list.append(G)

    out = [G_list]
    if return_encoded:
        out.append(adj_seq_array_list)
        out.append(node_features_seq_array_list)
    out = tuple(out)
    if len(out) == 1:
        out = out[0]
    return out
# ------------------------------------------------------------------------------
