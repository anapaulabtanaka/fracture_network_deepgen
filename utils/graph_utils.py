#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tools for Graph object (networkx).
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
import networkx

import matplotlib.pyplot as plt
import pyvista as pv

import os

# import torch
# import scipy

# # import from package 'geone'
# import geone as gn

# =============================================================================
# Utils for loading/saving graph(s) from/to files
# =============================================================================
# -----------------------------------------------------------------------------
def load_networkx_graph(
        dir, 
        name,
        suffix_nodes='_nodes.dat', 
        suffix_edges='_links.dat', 
        delimiter_nodes=' ',  
        delimiter_edges=' ',
        node_attrs=['pos'],
        node_attrs_ind=[(0, 1, 2)],
        nodet_attrs_type=['float'],
        start_id_at_0=True):
    """
    Loads a graph from two files (nodes an edges) and retrieves the graph as networkx.

    The graph is assumed to not have any isolated node, i.e. each node belongs to 
    an edge.

    Parameters
    ----------
    dir : str
        directory containing the files to be read
    
    name : str
        name of the graph to be read, the files

        - "`name``suffix_nodes`" : contains node data, i.e. \
        the list the attributes for each node \
        (one attribute per column, one node per row)
        - "`name``suffix_edges`" : contains the list of edges \
        (node ids of edge (2 columns))
    
    suffix_nodes : str, default: '_nodes.dat'
        suffix for file containing the node data
    
    suffix_edges : str, default: '_links.dat'
        suffix for file containing the edges (links) between nodes
    
    delimiter_nodes : str, default: ' '
        delimiter used in file for node data
    
    delimiter_edges : str, default ' '
        delimiter used in file for edges
    
    node_attrs : list of str, default: ['pos']
        name of the node attributes;
        (`None` or empty list means that there is no node attributes)
    
    node_attrs_ind : list of tuple, default: [(0, 1, 2)]
        tuple of column index-es of each node attribute        
        (index of column start at 0)
    
    node_attrs_type : list of type, default: ['float']
        type of node attributes
    
    start_id_at_0 : bool, default: `True`
        - if `True`: the node ids are converted to integers beginning at 0
        - if `False`: the node ids are not modified

    Returns
    -------
    G : `networkx.Graph`
        undirected graph, with node attributes
    """
    fname = 'load_networkx_graph'

    # Files
    filename_nodes = os.path.join(dir, f'{name}{suffix_nodes}')
    filename_edges = os.path.join(dir, f'{name}{suffix_edges}')

    if not os.path.isfile(filename_edges):
        print(f'ERROR ({fname}): File for edges does not exist')
        return None

    # Read edges (topology)
    data_edges = np.loadtxt(filename_edges, delimiter=delimiter_edges).astype(int)
    data_edges_list = list(map(tuple, data_edges))

    # Create graph and set topology
    G = networkx.Graph()
    G.add_nodes_from(np.unique(data_edges)) # ensure that all nodes are set in increasing order
    G.add_edges_from(data_edges_list)

     # Read and add node attributes
    if node_attrs is not None and len(node_attrs) > 0:
        if not os.path.isfile(filename_nodes):
            print(f'ERROR ({fname}): File for nodes does not exist')
            return None
        data_nodes = np.loadtxt(filename_nodes, delimiter=delimiter_nodes)
        for attr, attr_ind, attr_type in zip(node_attrs, node_attrs_ind, nodet_attrs_type):
            dict_attr = {ni:d.tolist() for ni, d in zip(G.nodes(), data_nodes[:, attr_ind].astype(attr_type))}
            networkx.set_node_attributes(G, dict_attr, attr)

    if start_id_at_0:
        # Start id from 0
        G = networkx.convert_node_labels_to_integers(G)
    
    return G
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def save_networkx_graph(
        G,
        dir, 
        name,
        suffix_nodes='_nodes.dat',
        suffix_edges='_links.dat', 
        delimiter_nodes=' ',
        delimiter_edges=' ',
        node_attrs=['pos'],
        fmt_nodes='%.10g',
        fmt_edges='%i'):
    """
    Saves a graph in two files (nodes an edges).

    Based on `numpy.savetxt`.

    Parameters
    ----------
    G : `networkx.Graph`
        undirected graph, node attributes
    
    dir : str
        directory containing the files to be written
    
    name : str
        name of the graph to be written, the files

        - "`name``suffix_nodes`" : contains node data, i.e. \
        the list the attributes for each node \
        (one attribute per column, one node per row)
        - "`name``suffix_edges`" : contains the list of edges \
        (node ids of edge (2 columns))
    
    suffix_nodes : str, default: '_nodes.dat'
        suffix for file containing the node data
    
    suffix_edges : str, default: '_links.dat'
        suffix for file containing the edges (links) between nodes
    
    delimiter_nodes : str, default ' '
        delimiter used in file for node data
    
    delimiter_edges : str, default ' '
        delimiter used in file for edges
    
    node_attrs : list of str, default: ['pos']
        name of the node attributes to be saved (written in file)
        (`None` or empty list means no node attributes)
    
    fmt_nodes : str, default '%.10g'
        format string for data attached to nodes (coordinates)
    
    fmt_edges : str, default '%i'
        format string for edges (node ids)
    """
    fname = 'save_networkx_graph'

    if not os.path.isdir(dir):
        print(f'ERROR ({fname}): Directory (where storing the files) not valid')
        return None

    # Files
    filename_nodes = os.path.join(dir, f'{name}{suffix_nodes}')
    filename_edges = os.path.join(dir, f'{name}{suffix_edges}')

    # Write edges (topology)
    data_edges = np.array(G.edges())
    np.savetxt(filename_edges, data_edges, delimiter=delimiter_edges, fmt=fmt_edges)
    
    # Write nodes attributes
    if node_attrs is not None:
        node_attrs = [attr for attr in node_attrs if attr in G.nodes[0].keys()]
        if len(node_attrs) > 0:
            data_nodes = np.hstack([list(networkx.get_node_attributes(G, attr).values()) for attr in node_attrs])
            np.savetxt(filename_nodes, data_nodes, delimiter=delimiter_nodes, fmt=fmt_nodes)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def load_networkx_graph_list(
        dir, 
        name,
        suffix_nodes='_nodes.dat', 
        suffix_edges='_links.dat', 
        delimiter_nodes=' ',  
        delimiter_edges=' ',
        node_attrs=['pos'],
        node_attrs_ind=[(0, 1, 2)],
        nodet_attrs_type=['float'],
        start_id_at_0=True):
    """
    Loads a list of graphs from two files (nodes an edges) and retrieves the graphs as networkx.

    Each graph is assumed to not have any isolated node, i.e. each node belongs to 
    an edge.

    Parameters
    ----------
    dir : str
        directory containing the files to be read
    
    name : str
        name of the graph to be read, the files

        - "`name``suffix_nodes`" : contains node data, i.e. \
        the list the attributes for each node, \
        preceeded by the graph index on each row \
        (one attribute per column, one node per row)
        - "`name``suffix_edges`" : contains the list of edges \
        preceeded by the graph index on each row

    suffix_nodes : str, default: '_nodes.dat'
        suffix for file containing the node data
    
    suffix_edges : str, default: '_links.dat'
        suffix for file containing the edges (links) between nodes
    
    delimiter_nodes : str, default ' '
        delimiter used in file for node data
    
    delimiter_edges : str, default ' '
        delimiter used in file for edges
    
    node_attrs : list of str, default: ['pos']
        name of the node attributes;
        (`None` or empty list means that there is no node attributes)
    
    node_attrs_ind : list of tuple, default: [(0, 1, 2)]
        tuple of column index-es of each node attribute,
        without accounting for graph index column        
        (index of column start at 0)
    
    node_attrs_type : list of type, default: ['float']
        type of node attributes
    
    start_id_at_0 : bool, default: `True`
        - if `True`: the node ids are converted to integers beginning at 0
        - if `False`: the node ids are not modified

    Returns
    -------
    G_list : list of `networkx.Graph`
        list of undirected graph, with node attributes
    """
    fname = 'load_networkx_graph_list'

    # Files
    filename_nodes = os.path.join(dir, f'{name}{suffix_nodes}')
    filename_edges = os.path.join(dir, f'{name}{suffix_edges}')

    if not os.path.isfile(filename_edges):
        print(f'ERROR ({fname}): File for edges does not exist')
        return None

    # Read edges (topology)
    data_edges = np.loadtxt(filename_edges, delimiter=delimiter_edges).astype(int)

    if node_attrs is not None and len(node_attrs) > 0:
        if not os.path.isfile(filename_nodes):
            print('fERROR ({fname}): File for nodes does not exist')
            return None
        # Read nodes attributes
        data_nodes = np.loadtxt(filename_nodes, delimiter=delimiter_nodes)
    else:
        data_nodes = None

    G_list = []
    n = max(data_edges[:, 0]) + 1 # number of graphs
    for i in range(n):
        # Create graph (i)
        data_edges_i = data_edges[data_edges[:, 0]==i, 1:]
        data_edges_list = list(map(tuple, data_edges_i))
        G = networkx.Graph()
        G.add_nodes_from(np.unique(data_edges_i)) # ensure that all nodes are set in increasing order
        G.add_edges_from(data_edges_list) # topology
        if data_nodes is not None:
            # Add node attributes
            for attr, attr_ind, attr_type in zip(node_attrs, node_attrs_ind, nodet_attrs_type):
                dd = data_nodes[data_nodes[:, 0]==i, 1:]
                dict_attr = {ni:d.tolist() for ni, d in zip(G.nodes(), dd[:, attr_ind].astype(attr_type))}
                networkx.set_node_attributes(G, dict_attr, attr)
        if start_id_at_0:
            # Start id from 0
            G = networkx.convert_node_labels_to_integers(G)
        G_list.append(G)
    
    return G_list
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def save_networkx_graph_list(
        G_list,
        dir, 
        name,
        suffix_nodes='_nodes.dat',
        suffix_edges='_links.dat', 
        delimiter_nodes=' ',
        delimiter_edges=' ',
        node_attrs=['pos'],
        fmt_nodes='%.10g',
        fmt_edges='%i'):
    """
    Saves a list of graph in two files (nodes an edges).

    Based on `numpy.savetxt`.

    Parameters
    ----------
    G_list : list of `networkx.Graph`
        list of undirected graphs, with node attributes
    
    dir : str
        directory containing the files to be written
    
    name : str
        name of the graph to be written, the files

        - "`name``suffix_nodes`" : contains node data, i.e. \
        the list the attributes for each node, \
        preceeded by the graph index on each row \
        (one attribute per column, one node per row)
        - "`name``suffix_edges`" : contains the list of edges \
        preceeded by the graph index on each row
    
    suffix_nodes : str, default: '_nodes.dat'
        suffix for file containing the node data
    
    suffix_edges : str, default: '_links.dat'
        suffix for file containing the edges (links) between nodes
    
    delimiter_nodes : str, default ' '
        delimiter used in file for node data
    
    delimiter_edges : str, default ' '
        delimiter used in file for edges
    
    node_attrs : list of str, default: ['pos']
        name of the node attributes to be saved (written in file)
        (`None` or empty list means no node attributes)
    
    fmt_nodes : str, default '%.10g'
        format string for data attached to nodes (coordinates)
    
    fmt_edges : str, default '%i'
        format string for edges (node ids)
    """
    fname = 'save_networkx_graph_list'
    
    if not os.path.isdir(dir):
        print(f'ERROR ({fname}): Directory (where storing the files) not valid')
        return None

    # Files
    filename_nodes = os.path.join(dir, f'{name}{suffix_nodes}')
    filename_edges = os.path.join(dir, f'{name}{suffix_edges}')

    # Write edges (topology)
    with open(filename_edges, 'w') as f:
        for i, G in enumerate(G_list):
            data_edges = np.array(G.edges())
            data_edges = np.hstack((np.full((data_edges.shape[0], 1), i), data_edges))
            np.savetxt(f, data_edges, delimiter=delimiter_edges, fmt=fmt_edges)
    
    # Write nodes attributes
    if node_attrs is not None:
        node_attrs = [attr for attr in node_attrs if attr in G_list[0].nodes[0].keys()]
        if len(node_attrs) > 0:
            data_nodes = np.hstack([list(networkx.get_node_attributes(G, attr).values()) for attr in node_attrs])
            np.savetxt(filename_nodes, data_nodes, delimiter=delimiter_nodes, fmt=fmt_nodes)
            with open(filename_nodes, 'w') as f:
                for i, G in enumerate(G_list):
                    data_nodes = np.hstack([list(networkx.get_node_attributes(G, attr).values()) for attr in node_attrs])
                    data_nodes = np.hstack((np.full((data_nodes.shape[0], 1), i), data_nodes))
                    np.savetxt(f, data_nodes, delimiter=delimiter_nodes, fmt=fmt_nodes)
# -----------------------------------------------------------------------------

# =============================================================================
# Utils for comparing two graphs
# =============================================================================
# -----------------------------------------------------------------------------
def compare_graph(G1, G2, node_attrs=None, verbose=0, **kwargs):
    """
    Compares two graphs.

    Parameters
    ----------
    G1 : `networkx.Graph`
        first graph
    
    G2 : `networkx.Graph`
        second graph
    
    node_attrs : list of str, optional
        name of the node attributes to be compared
    
    verbose : int, default: 0
        verbosisty (higher implies more printing)
    
    kwargs: dict
        keyword arguments passed to the function `numpy.allclose`, when
        comparing node attributes, e.g. kwargs = dict(rtol=1.e-5, atol=1.e-8)
        (default values here for `numpy.allclose`)

    Returns
    -------
    check_list : list of bools
        list of bools, where each element is the result of comparison of

        - number of nodes
        - number of edges
        - adjacency matrix
        - each node attribute to be compared
    """
    check_list = []

    ok_n = G1.number_of_nodes() == G2.number_of_nodes()
    check_list.append(ok_n)
    if verbose > 0:
        print(f'... {ok_n} - number of nodes')

    ok_e = G1.number_of_edges() == G2.number_of_edges()
    check_list.append(ok_e)
    if verbose > 0:
        print(f'... {ok_e} - number of edges')

    if ok_n and ok_e:
        ok = np.all(networkx.adjacency_matrix(G1).toarray()==networkx.adjacency_matrix(G2).toarray())
        check_list.append(ok)
        if verbose > 0:
            print(f'... {ok} - adj. mat.')

    if ok_n and node_attrs is not None:
        for attr in node_attrs:
            x1 = np.asarray(list(networkx.get_node_attributes(G1, attr).values()))
            x2 = np.asarray(list(networkx.get_node_attributes(G2, attr).values()))
            ok = x1.shape == x2.shape
            check_list.append(ok)
            if verbose > 0:
                print(f'... {ok} - attribute {attr}, array shape')
            if not ok:
                check_list.append(ok)
                if verbose > 0:
                    print(f'... False (NOT CHECKED) - attribute {attr}, array values')
            else:
                ok = np.allclose(x1, x2, **kwargs)
                check_list.append(ok)
                if verbose > 0:
                    print(f'... {ok} - attribute {attr}, array values')
                    if verbose > 1:
                        x = x1 - x2
                        print(f'... {x.min()}, {x.max()} - min max of diff attribute {attr}')

    return check_list
# -----------------------------------------------------------------------------

# =============================================================================
# Utils - basic manipulation of graph
# =============================================================================
# -----------------------------------------------------------------------------
def remove_node_attribute(G, attr):
    """
    Removes a node attribute.

    Parameters
    ----------
    G : networkx.Graph object
        graph
    
    attr : str
        name of the node attribute to be removed
    
    Returns
    -------
    G : networkx.Graph object
        graph in output

    Notes
    -----
    Inplace operation: `G` will be modified.
    """
    for n in G.nodes():
        if attr in G.nodes[n].keys():
            del G.nodes[n][attr]
    
    return G
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def remove_all_node_attributes_but(G, attr_to_keep):
    """
    Removes all node attributes not in the given list.

    Parameters
    ----------
    G : networkx.Graph object
        graph
    
    attr : list of str
        names of the node attribute to be kept
    
    Returns
    -------
    G : networkx.Graph object
        graph in output

    Notes
    -----
    Inplace operation: `G` will be modified.
    """
    for n in G.nodes():
        attr_to_remove = [attr_key for attr_key in G.nodes[n].keys() if attr_key not in attr_to_keep]
        for attr_key in attr_to_remove:
            del G.nodes[n][attr_key]

    return G
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def rename_node_attribute(G, attr, attr_new):
    """
    Renames a node attribute (and delete the original attribute).

    Parameters
    ----------
    G : networkx.Graph object
        graph
    
    attr : str
        name of the node attribute to be changed
    
    attr_new : str
        new name for the node attribute to be changed
    
    Returns
    -------
    G : networkx.Graph object
        graph with renamed node attribute

    Notes
    -----
    Inplace operation: `G` will be modified.
    """
    for n in G.nodes():
        if attr in G.nodes[n].keys():
            G.nodes[n][attr_new] = G.nodes[n][attr]
            del G.nodes[n][attr]
    
    # Note: 
    # networkx.set_node_attributes(G, networkx.get_node_attributes(G, attr), attr_new)
    # does not remove the original attribute
    return G
# -----------------------------------------------------------------------------

# # Notes:
# # -----
# # To convert an undirected graph with node position (attribute 'pos'):
# # - from networkx to torch_geometric
# G_pygeom = torch_geometric.utils.from_networkx(G)
# # - from torch_geometric to networkx
# G = torch_geometric.utils.to_networkx(G_pygeom, to_undirected=True, node_attrs=['pos'])

# =============================================================================
# Utils for plotting graph in 3D with py vista
# =============================================================================
# -----------------------------------------------------------------------------
def get_mesh_from_graph(G, start_id=0, pos_attr='pos'):
    """
    Get mesh of a graph (for plotting with pyvista).

    The graph `G` in input is assumed to have nodes localized in 3D space.

    Parameters
    ----------
    G : networkx.Graph object
        graph, with nodes with integer id, and 3D position as attribute
    
    start_id : int, default: 0
        start node id
    
    pos_attr : str, default: 'pos'
        name of the attribute attached to nodes given the position as a 
        sequence of 2 or 3 floats; if position are in 2D, a coordinate 
        of zero is set in the 3rd dimension

    Returns
    -------
    mesh : pyvista.PolyData object
        mesh with points (node of the graph `G`) and lines (edges of 
        the graph `G`), that can be plotted with pyvista
    """
    # points: 2d-array of shape (n_nodes, 3), row i is the position in 3D of the node id i
    points = np.array([G.nodes[ni][pos_attr] for ni in G.nodes()])
    if points.shape[1] == 2: # 2D
        points = np.insert(points, 2, 0.0, axis=1) # add 0.0 as 3-rd coordinate (z)

    # lines: 2d-array of shape (n_edges, 3), row i is [2, j0, j1] where (j0, j1) is an edge btw node id j0 and node id j1
    lines = np.insert(np.asarray(G.edges())-start_id, 0, 2, axis=1)

    mesh = pv.PolyData(points, lines=lines.ravel())

    return mesh
# -----------------------------------------------------------------------------

# =============================================================================
# Utils for simple computation on graph
# =============================================================================
# -----------------------------------------------------------------------------
def reorder_nodes(G, seq):
    """
    Reorder graph nodes according to sequence of visited nodes.

    Parameters
    ----------
    G : networkx.Graph object
        graph, with nodes labels assumed to be integers from 0 (0, 1, 2, ...)
    
    seq : sequence of ints,
        permutation of [0, ..., n_nodes-1] where n_nodes is the number of 
        nodes in `G`, sequence of visited nodes, i.e. the "node id" seq[j]
        in input becomes the "new node id" j in output
    
    Returns
    -------
    G_new: networkx.Graph object
        new graph with reordered nodes, according to `seq`

    Notes
    -----
    This function reorders the nodes and computes a new graph, not just 
    relabeling the nodes. Moreover, the node attributes are transmitted in
    the new graph.
    """
    # Compute the new graph according to sequence of visited nodes
    # (The new graph is built from the permuted adjacency matrix of the original)
    G_new = networkx.from_scipy_sparse_array(networkx.adjacency_matrix(G, seq))

    # Attach the attributes of the nodes from the original graph
    for i in range(G.number_of_nodes()):
        for k, v in G.nodes[seq[i]].items():
            G_new.nodes[i][k] = v

    return G_new
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def get_connected_nodes(G, id, seq, seq_inv=None, prev_nodes_only=False):
    """
    Gets connected nodes in a graph.
    
    Consider a graph `G` and "new node ids" set according to a sequence `seq` of
    visited nodes (permutation of [0, ..., n_nodes - 1], where n_nodes is the 
    number of nodes in `G`), i.e. new node id i is the node id seq[i] in the 
    original graph. 
    
    This function retrieves the list of new ids of nodes connected (with an edge)
    to the nodes of new id `id`, i.e. the row indices (resp. column indices) of 
    the ones in the column (resp. row) `id` of the adjacency matrix wrt to 
    sequence `seq`.

    Parameters
    ----------
    G : networkx.Graph object
        graph 
    
    id : int
        new node id (integer in [0, ..., n_nodes-1]), wrt sequence of visited 
        nodes `seq`
    
    seq : sequence of ints,
        permutation of [0, ..., n_nodes-1] where n_nodes is the number of 
        nodes in `G`, sequence of visited nodes, i.e. the "node id" seq[j]
        in input becomes the "new node id" j in output
    
    seq_inv:  sequence of ints, optional
        inverse permutation of `seq`, i.e. `seq_inv = np.argsort(seq)` s.t. 
        seq[seq_inv[i]] = seq_inv[seq[i]] = i, for all i;
        by default (`None`): automatically computed
    
    prev_nodes_only : bool, default: `False`
        - if `True`: only the connected nodes with new id less than `id` are returned 
        - if `False`: all the connected nodes with new id less than `id` are returned

    Returns
    -------
    connected_ids: sequence of int(s)
        list of new ids of nodes to node of new id `id` (only new id less than
        `id` if `prev_nodes_only=True`)
    """
    # New adjacency matrix in csr (compressed-sparse-row) format:
    #   adj_mat_new_csr = networkx.adjacency_matrix(G, seq)
    #
    # With M the new adjacency matrix
    #   n   : the order of M, i.e. the number of nodes in the graph
    #   nnz : number of non-zeros entries in the matrix
    # we have:
    #   adj_mat_new_csr.indptr : 1d-array of shape(n+1, ) of ints: index pointer to rows
    #   adj_mat_new_csr.indices: 1d-array of shape(nnz, ) of ints containing column indices
    #   adj_mat_new_csr.data   : 1d-array of shape(nnz, ) of ints containing values
    # with
    #   adj_mat_new_csr.indptr[0] = 0 < adj_mat_new_csr.indptr[1] < ... < adj_mat_new_csr.indptr[n] = nnz
    #   for adj_mat_new_csr.indptr[i] <= k < adj_mat_new_csr.indptr[i]:
    #       M[i, adj_mat_new_csr.indices[k]] = adj_mat_new_csr.data[k]
    #
    # Note: 
    #    adj_mat_new_csr.indices[adj_mat_new_csr.indptr[i]:adj_mat_new_csr.indptr[i+1]]: 
    #       sequence of column indices for row i, sorted in increasing order

    if seq_inv is None:
        seq_inv = np.argsort(seq)

    connected_ids = np.sort(seq_inv[np.asarray(list(G.edges(seq[id])))[:, 1]])

    # # Equivalently:
    # adj_mat_new_csr = networkx.adjacency_matrix(G, seq)
    # connected_ids = adj_mat_new_csr.indices[adj_mat_new_csr.indptr[id]:adj_mat_new_csr.indptr[id+1]]
    
    if prev_nodes_only:
        connected_ids = connected_ids[connected_ids < id]

    return connected_ids
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def get_bfs_sequence(G, start_id, depth_limit=None):
    """
    Get breadth-first-search (bfs) sequence.

    The bfs sequence is a sequence of graph nodes (ids) built as follows:

    - start with the bfs sequence of one graph node: [id]
    - then, iteratively find the first graph node id in the bfs sequence that \
    has at least one neighbor (i.e. node connected with an edge) not yet in \
    the bfs sequence, and append all its neighbors to the bfs sequence

    Parameters
    ----------
    G : networkx.Graph object
        graph
    
    start_id : int (or str)
        node graph id starting the sequence
    
    depth_limit : int, optional
        maximal search depth (see function `networkx.bfs_successors`)
    
    Returns
    -------
    bfs_seq : 1d-array
        bfs sequence
    """
    # Flag the ids: True if not yet integrated in the bfs sequence
    id_flag = np.ones(G.number_of_nodes(), dtype='bool')

    bfs_seq = [start_id]
    id_flag[start_id] = False

    ids_to_treat = [start_id]
    while len(ids_to_treat) > 0:
        next_ids = np.array([], dtype='int')
        for id in ids_to_treat:
            neighbors_id = np.asarray(list(G.edges(id)))[:, 1]
            neighbors_id = neighbors_id[id_flag[neighbors_id]]
            next_ids = np.hstack((next_ids, neighbors_id))
            id_flag[neighbors_id] = False
        bfs_seq = np.hstack((bfs_seq, next_ids))
        ids_to_treat = next_ids

    return bfs_seq

    # # Equivalent:
    # bfs_succ_dict = dict(networkx.bfs_successors(G, start_id, depth_limit=depth_limit))

    # bfs_seq = [start_id]
    
    # ids_to_treat = [start_id]
    # while len(ids_to_treat) > 0:
    #     next_ids = []
    #     for id in ids_to_treat:
    #         neighbors_id = bfs_succ_dict.get(id)
    #         if neighbors_id is not None:
    #             next_ids = next_ids + neighbors_id
    #     bfs_seq = bfs_seq + next_ids
    #     ids_to_treat = next_ids

    # return np.asarray(bfs_seq)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def csr_array_bw(S):
    """
    Computes the bandwidth of a matrix in csr (compressed-sparse-row) format.
    
    The bandwidth of a matrix :math:`M=(m_{ij})` is defined as 
    :math:`bw = \max\{|i-j| : m_{ij} \\neq 0\}`
    (i.e. a diagonal matrix has a bandwidth of 0 with this definition).

    Parameters
    ----------
    S : scipy.sparse.csr_array
        matrix in csr format 
        
    Returns
    -------
    bw: int
        bandwidth of `S`
    """
    if S.nnz == 0:
        # empty matrix
        return 0
    
    bw = np.max(np.hstack([np.abs(i - S.indices[S.indptr[i]:S.indptr[i+1]]) for i in range(S.shape[0])]))

    return bw
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def adj_mat_bw(G, seq=None):
    """
    Computes the bandwidth of the adjacency matrix of a graph G wrt sequence of visited nodes.
    
    Consider a graph `G` and "new node ids" set according to a sequence `seq` of
    visited nodes (permutation of [0, ..., n_nodes - 1], where n_nodes is the 
    number of nodes in `G`), i.e. new node id i is the node id seq[i] in the 
    original graph.
    
    This function computes the bandwidth of the adjacency matrix of the graph 
    for the new node ids.
    
    The bandwidth of a matrix :math:`M=(m_{ij})` is defined as 
    :math:`bw = \max\{|i-j| : m_{ij} \\neq 0\}`
    (i.e. a diagonal matrix has a bandwidth of 0 with this definition).

    Parameters
    ----------
    G : networkx.Graph object
        graph 
    
    seq : sequence of ints, optional
        permutation of [0, ..., n_nodes-1] where n_nodes is the number of 
        nodes in `G`, sequence of visited nodes, i.e. the "node id" seq[j]
        in input becomes the "new node id" j in output
        
    Returns
    -------
    bw: int
        bandwidth of adjacency matrix of graph `G` wrt sequence of visited 
        nodes `seq`
    """
   
    # # Equivalently:
    # seq_inv = np.argsort(seq)
    # bw = 0
    # for i in range(G.number_of_nodes()):
    #     jcol = seq_inv[np.asarray(list(G.edges(seq[i])))[:, 1]]
    #     if jcol.size > 0:
    #         bw = max(bw, np.max(np.abs(i - jcol)))

    # # Equivalently:
    # seq_inv = np.argsort(seq)
    # bw = 0
    # for i in range(G.number_of_nodes()):
    #     jcol = get_connected_nodes(G, i, seq, seq_inv, prev_nodes_only=True)
    #     if jcol.size > 0:
    #         bw = max(bw, np.max(i - jcol))

    # # Equivalently:
    # bw = 0
    # adj_mat_new_csr = networkx.adjacency_matrix(G, seq)
    # for i in range(G.number_of_nodes()):
    #     bw = max(bw, np.max(np.abs(i - adj_mat_new_csr.indices[adj_mat_new_csr.indptr[i]:adj_mat_new_csr.indptr[i+1]])))
        
    bw = csr_array_bw(networkx.adjacency_matrix(G, nodelist=seq))

    return bw
# -----------------------------------------------------------------------------

# =============================================================================
# Utils for extracting graph
# =============================================================================
# -----------------------------------------------------------------------------
def extract_subgraph_from_bfs(G, n_nodes, start_id=0, randomize_nodes=True, seed=None):
    """
    Extracts a subgraph.

    Starting from the input graph `G` (with integers starting from 0 as node numbering):
    
    - its nodes are randomized (if `randomize_nodes=True`)
    - the BFS (breadth-first-search) sequence is computed from the start \
    node id `start_id`
    - the subgraph containing the first `n_nodes` of the BFS sequence is extracted

    Parameters
    ----------
    G : `networkx.Graph`
        graph
    
    n_nodes : int
        number of nodes to be extracted
    
    start_id : int, default: 0
        starting id for the BFS sequence
    
    randomize_nodes : bool, default: `True`
        if `True`: nodes (numbering) of `G` are first randomized
    
    seed : int, optional
        seed number (used if `randomize_nodes=True`)
    """
    attr_key_list = G.nodes[0].keys()
    attr_list = [np.array(list(networkx.get_node_attributes(G, attr_key).values())) for attr_key in attr_key_list]

    G_out = G.copy()

    if randomize_nodes:
        if seed is not None:
            np.random.seed(seed)
            # torch.random.manual_seed(seed)
        seq = np.random.permutation(G_out.number_of_nodes())
        # seq = torch.randperm(G.number_of_nodes()).numpy()
        adj_mat_csr = networkx.adjacency_matrix(G_out, seq)
        G_out = networkx.from_scipy_sparse_array(adj_mat_csr)
        attr_list = [attr[seq] for attr in attr_list]
        
    seq = get_bfs_sequence(G_out, start_id)
    adj_mat_csr = networkx.adjacency_matrix(G_out, seq)
    G_out = networkx.from_scipy_sparse_array(adj_mat_csr)
    
    nn = min(G_out.number_of_nodes(), n_nodes)
    G_out = G_out.subgraph(np.arange(nn)).copy()

    for attr_key, attr in zip(attr_key_list, attr_list):
        dict_attr = {i:v.tolist() for i, v in enumerate(attr[seq[:nn]])}
        networkx.set_node_attributes(G_out, dict_attr, attr_key)

    return G_out
# -----------------------------------------------------------------------------

# =============================================================================
# Utils to deal with node features
# =============================================================================

# -----------------------------------------------------------------------------
def extract_graph_node_features_indices(G, attr, indices, inplace=True):
    """
    Extracts some indices from node features in a graph.

    Parameters
    ----------
    G : `networkx.Graph`
        graph
    
    attr : str
        name of the considered feature attribute
    
    indices : list (or tuple) of ints
        indices to be kept from attribute `attr`
    
    inplace : bool, default: `True`
        - if `True`: operation is applied inplace (`G` is modified)
        - if `False`: operation is not applied inplace (`G` is not modified)

    Returns
    -------
    G_out : `networkx.Graph`
        output graph
    
    Examples
    --------
    To extract two first coordinates of attribute 'pos'::
    
        G = extract_graph_node_features_indices(G, 'pos', (0, 1))
    """
    if inplace:
        G_out = G
    else:
        G_out = G.copy()

    feat_array = np.asarray(list(networkx.get_node_attributes(G_out, attr).values()))[:, indices]
    dict_feat = {ni: feat.tolist() for ni, feat in zip(G_out.nodes(), feat_array)}
    networkx.set_node_attributes(G_out, dict_feat, attr)

    return G_out
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def rescale_graph_node_features(G, attr, vmin, vmax, inplace=True, return_initial_min_max=False):
    """
    Rescales node features in a graph.

    Parameters
    ----------
    G : `networkx.Graph`
        graph
    
    attr : str
        name of the considered feature attribute
    
    vmin, vmax : floats or sequence of floats
        target interval for each feature index
        if float, the value is repeated for each index
    
    inplace : bool, default: `True`
        - if `True`: operation is applied inplace (`G` is modified)
        - if `False`: operation is not applied inplace (`G` is not modified)
    
    return_initial_min_max : bool, default: `False`
        if `True`: initial min and max values for each feature index are returned
    
    Returns
    -------
    G_out : `networkx.Graph`
        output graph
    
    initial_min : 1d-array of floats, optional
        minimal value of each feature index, returned if `return_initial_min_max=True`
    
    initial_max : 1d-array of floats, optional
        maximal value of each feature index, returned if `return_initial_min_max=True`

    Examples
    --------
    ::

        # To rescale coordinates of attribute 'pos' (3D) in [-1, 1] along each axis, and 
        # keep track of initial min and max:
        G, p_min, p_max = rescale_graph_node_features(G, 'pos', -1, 1, return_initial_min_max=True)
        # To rescale back to original values:
        G = rescale_graph_node_features(G, 'pos', p_min, p_max)
    """
    fname='rescale_graph_node_features'

    if inplace:
        G_out = G
    else:
        G_out = G.copy()

    feat_array = np.asarray(list(networkx.get_node_attributes(G_out, attr).values()))

    dim = feat_array.shape[1]
    vmin = np.asarray(vmin, dtype='float').reshape(-1)
    if len(vmin) == 1:
        vmin = np.repeat(vmin, dim)
    elif len(vmin) != dim:
        print(f"ERROR ({fname}): `vmin` not valid")
        if return_initial_min_max:
            return None, None, None
        else:
            return None
    vmax = np.asarray(vmax, dtype='float').reshape(-1)
    if len(vmax) == 1:
        vmax = np.repeat(vmax, dim)
    elif len(vmax) != dim:
        print(f"ERROR ({fname}): `vmax` not valid")
        if return_initial_min_max:
            return None, None, None
        else:
            return None

    feat_min = feat_array.min(axis=0)
    feat_max = feat_array.max(axis=0)
    feat_array = vmin + (feat_array - feat_min)/(feat_max - feat_min) * (vmax - vmin)
    dict_feat = {ni: feat.tolist() for ni, feat in zip(G_out.nodes(), feat_array)}
    networkx.set_node_attributes(G_out, dict_feat, attr)

    if return_initial_min_max:
        return G_out, feat_min, feat_max
    else:
        return G_out
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def centralize_graph_node_features(G, attr, inplace=True, return_initial_mean=False):
    """
    Shifts node features in a graph, such that their mean is zero.

    Parameters
    ----------
    G : `networkx.Graph`
        graph
    
    attr : str
        name of the considered feature attribute
    
    inplace : bool, default: `True`
        - if `True`: operation is applied inplace (`G` is modified)
        - if `False`: operation is not applied inplace (`G` is not modified)
    
    return_initial_mean : bool, default: `False`
        if `True`: initial mean values for each feature index are returned
    
    Returns
    -------
    G_out : `networkx.Graph`
        output graph
    
    initial_mean : 1d-array of floats, optional
        mean value of each feature index, returned if `return_initial_mean=True`

    Examples
    --------
    ::
    
        # To centralize coordinates of attribute 'pos' (3D), so that the mean is at origin, and 
        # keep track of initial mean:
        G, p_mean = centralize_graph_node_features(G, 'pos', return_initial_mean=True)
        # To rescale back to original values:
        G = shift_graph_node_features(G, 'pos', p_mean)
    """
    fname='centralize_graph_node_features'

    if inplace:
        G_out = G
    else:
        G_out = G.copy()

    feat_array = np.asarray(list(networkx.get_node_attributes(G_out, attr).values()))

    feat_mean = feat_array.mean(axis=0)
    feat_array = feat_array - feat_mean
    dict_feat = {ni: feat.tolist() for ni, feat in zip(G_out.nodes(), feat_array)}
    networkx.set_node_attributes(G_out, dict_feat, attr)

    if return_initial_mean:
        return G_out, feat_mean
    else:
        return G_out
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def shift_graph_node_features(G, attr, vshift, inplace=True):
    """
    Shifts node features in a graph.

    Parameters
    ----------
    G : `networkx.Graph`
        graph
    
    attr : str
        name of the considered feature attribute
    
    vshift : floats or sequence of floats
        shift vector
        if float, the value is repeated for each index
    
    inplace : bool, default: `True`
        - if `True`: operation is applied inplace (`G` is modified)
        - if `False`: operation is not applied inplace (`G` is not modified)
    
    Returns
    -------
    G_out : `networkx.Graph`
        output graph
    """
    fname='shift_graph_node_features'

    if inplace:
        G_out = G
    else:
        G_out = G.copy()

    feat_array = np.asarray(list(networkx.get_node_attributes(G_out, attr).values()))

    dim = feat_array.shape[1]
    vshift = np.asarray(vshift, dtype='float').reshape(-1)
    if len(vshift) == 1:
        vshift = np.repeat(vshift, dim)
    elif len(vshift) != dim:
        print(f"ERROR ({fname}): `vshift` not valid")
        return None

    feat_array = feat_array + vshift
    dict_feat = {ni: feat.tolist() for ni, feat in zip(G_out.nodes(), feat_array)}
    networkx.set_node_attributes(G_out, dict_feat, attr)

    return G_out
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def multiply_graph_node_features(G, attr, vmul, inplace=True):
    """
    Multiplies node features in a graph.

    Parameters
    ----------
    G : `networkx.Graph`
        graph
    
    attr : str
        name of the considered feature attribute
    
    vmul : floats or sequence of floats, or 2d array of floats
        - if 2d array: 
            multiplication matrix (of shape (n, m), where n is 
            then number of original node features, and m is the 
            number of new node features, after matrix multiplication:
            "node_features * vmul"  (i..e node_features in row vectors)
        - otherwise: 
            multiplication factors
            if float, the value is repeated for each index
    
    inplace : bool, default: `True`
        - if `True`: operation is applied inplace (`G` is modified)
        - if `False`: operation is not applied inplace (`G` is not modified)
    
    Returns
    -------
    G_out : `networkx.Graph`
        output graph
    """
    fname='multiply_graph_node_features'

    if inplace:
        G_out = G
    else:
        G_out = G.copy()

    feat_array = np.asarray(list(networkx.get_node_attributes(G_out, attr).values()))

    dim = feat_array.shape[1]
    if isinstance(vmul, np.ndarray) and np.ndim(vmul) == 2:
        if vmul.shape[0] != dim:
            print(f"ERROR ({fname}): `vmul` (matrix) not valid")
            return None
        feat_array = np.dot(feat_array, vmul)
    else:
        vmul = np.asarray(vmul, dtype='float').reshape(-1)
        if len(vmul) == 1:
            vmul = np.repeat(vmul, dim)
        elif len(vmul) != dim:
            print(f"ERROR ({fname}): `vmul` not valid")
            return None
        feat_array = feat_array * vmul

    dict_feat = {ni: feat.tolist() for ni, feat in zip(G_out.nodes(), feat_array)}
    networkx.set_node_attributes(G_out, dict_feat, attr)

    return G_out
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def pca_graph_node_features(G, attr, normalize=False, inplace=True, return_initial_mean_and_pca=False):
    """
    Applies PCA to node features of a graph.

    The node features are expressed in the PCA system, centered at the mean of 
    the original node features; moreoever, normalization (wrt. standard deviation along 
    principal axes) is done optionally.

    Note that the principal axes are sorted according the corresponding standard
    deviation in descending order.

    Parameters
    ----------
    G : `networkx.Graph`
        graph
    
    attr : str
        name of the considered feature attribute
    
    normalize : bool, default: `False`
        - if `True`: new features are normalized by the standard deviation along principal axes
        - if `False`: new features are not normalized
    
    inplace : bool, default: `True`
        - if `True`: operation is applied inplace (`G` is modified)
        - if `False`: operation is not applied inplace (`G` is not modified)
    
    return_initial_mean_and_pca : bool, default: `False`
        if `True`: initial mean values for each feature index, and PCA system 
        axes and standard deviation along each axis are returned

    Returns
    -------
    G_out : `networkx.Graph`
        output graph
    
    initial_mean : 1d-array of floats, optional
        mean value of each feature index, returned if `return_initial_mean_and_pca=True`
    
    pca_axes : 2d numpy array, optional
        PCA system axes, i.e. matrix whose columns are the principal axes (of norm 1), 
        returned if `return_initial_mean_and_pca=True`
    
    pca_std : 1d numpy array, optional
        standard deviation along PCA system axes, returned if 
        `return_initial_mean_and_pca=True`

    Examples
    --------
    Express coordinates of attribute 'pos' (3D) in the PCA axes system with normalization, 
    and keep track of transformation::

        # Do PCA
        G, p_mean, pca_axes, pca_std = pca_graph_node_features(
            G, 'pos', normalize=True, return_initial_mean_and_pca=True)
        # Back-transform to original values:
        G = multiply_graph_node_features(G, 'pos', pca_std)
        G = multiply_graph_node_features(G, 'pos', pca_axes.T)
        G = shift_graph_node_features(G, 'pos', p_mean)
    
    """
    # fname='pca_graph_node_features'

    if inplace:
        G_out = G
    else:
        G_out = G.copy()

    feat_array = np.asarray(list(networkx.get_node_attributes(G_out, attr).values()))

    # Covariance matrix of node features
    cov_mat = np.cov(feat_array.T)

    # Diagonalization
    d, smat = np.linalg.eig(cov_mat)
    # -> d: (1d-array) eigen values, variances along prinicpal axes
    # -> smat: matrix whose columns are the eigen vectors (of norm 1), i.e. principal axes
    d = np.sqrt(d) # std along principal axes

    # sort std in descending order
    ind = np.argsort(d)[::-1]
    smat = smat[:,ind]
    d = d[ind]

    # Express features in PCA system
    feat_mean = feat_array.mean(axis=0)
    feat_array = (feat_array-feat_mean).dot(smat)
    if normalize:
        # Normalize according to std
        feat_array = feat_array/d

    dict_feat = {ni: feat.tolist() for ni, feat in zip(G_out.nodes(), feat_array)}
    networkx.set_node_attributes(G_out, dict_feat, attr)

    if return_initial_mean_and_pca:
        return G_out, feat_mean, smat, d
    else:
        return G_out
# -----------------------------------------------------------------------------


###### OLD #####
# # =============================================================================
# # Utils for plotting graph
# # =============================================================================
# # -----------------------------------------------------------------------------
# def plot_graph_2d(G, pos_attr='pos', attr=None, attr_ind=0, with_labels=False, show_colorbar=True, **kwargs):
#     """
#     Plot a graph in 2D, in the current axis figure.

#     The graph is assumed to node attributes given the spatial position.

#     Parameters
#     ----------
#     G : networkx.Graph object
#         graph
#     pos_attr : str
#         name of the node attribute given the position of the nodes 
#     attr : str, optional
#         name of the node attribute to be plotted
#     attr_ind : int, default: 0
#         index of the attribute to be plotted (used if attr is not `None`)
#     with_labels : bool, default: False
#         indicates if the node id is plotted (`True`) or not (`False`)
#     show_colorbar : bool, default: True
#         indicates if the color bar is displayed (`True`) or not (`False`);
#         used if `attr` is not `None`

#     kwargs : dict
#         additional arguments, possible keys, values : 
        
#         - 'node_shape':'o',         # node symbol
#         - 'edgecolors':'green',     # node border color
#         - 'node_size':100,          # node size
#         - 'linewidths':3.2,         # node border width
#         - 'edge_color':'pink',      # edge color
#         - 'width':3.0,              # edge width
#         - 'font_size':9,            # font size (labels)
#         - 'font_color':'red',       # font color (labels)
#         - 'node_color':'lightblue', # node color (used if attr=None)
#         - 'cmap':'plasma'           # color map (used if attr is not None) 
#         - 'vmin':vmin               # min value for color map (used if attr is not None) 
#         - 'vmax':vmax               # max value for color map (used if attr is not None) 
#     """
#     # fname = 'plot_graph_2d'
    
#     dict_pos2d = {i: p[:2] for i, p in networkx.get_node_attributes(G, pos_attr).items()}

#     if attr is None:
#         # remove unappropriate keyword arguments
#         for k in ('cmap', 'vmin', 'vmax'):
#             if k in kwargs.keys():
#                 del kwargs[k]
#         networkx.draw(G, pos=dict_pos2d, with_labels=with_labels, **kwargs)
    
#     else:
#         # get value of the attribute (and index) to plot        
#         v = np.asarray(list(networkx.get_node_attributes(G, attr).values()))[:, attr_ind]

#         # get keyword arguments for edge and remove them from kwargs
#         if 'edge_color' in kwargs.keys():
#             edge_color = kwargs['edge_color']
#             del kwargs['edge_color']
#         else:
#             edge_color = None

#         if 'width' in kwargs.keys():
#             width = kwargs['width']
#             del kwargs['width']
#         else:
#             width = None

#         _ = networkx.draw_networkx_edges(G, pos=dict_pos2d, edge_color=edge_color, width=width)

#         # get keyword arguments for labels and remove them from kwargs
#         if 'font_color' in kwargs.keys():
#             font_color = kwargs['font_color']
#             del kwargs['font_color']
#         else:
#             font_color = None

#         if 'font_size' in kwargs.keys():
#             font_size = kwargs['font_size']
#             del kwargs['font_size']
#         else:
#             font_size = None

#         # remove unappropriate keyword arguments
#         for k in ('node_color',):
#             if k in kwargs.keys():
#                 del kwargs[k]

#         im_nodes = networkx.draw_networkx_nodes(G, pos=dict_pos2d, node_color=list(v), **kwargs)

#         if with_labels:
#             for ni in G.nodes():
#                 plt.text(*G.nodes[ni][pos_attr][:2], str(ni), 
#                         horizontalalignment='center',
#                         verticalalignment='center',
#                         color=font_color, fontsize=font_size
#                     )
#         if show_colorbar:
#             plt.colorbar(im_nodes)

# # -----------------------------------------------------------------------------
# def plot_graph_3d(G, attr=None, attr_ind=0, attr_label=None, with_labels=False, 
#                   plotter=None,
#                   show_scalar_bar=True,
#                   kwargs_edges=None, kwargs_pts=None, kwargs_scalar_bar=None, kwargs_pts_labels=None, 
#                   cpos=None):
#     """
#     Plot a graph in 3D (based on `pyvista`).

#     The graph is assumed to node attributes given the spatial position.

#     Parameters
#     ----------
#     G : networkx.Graph object
#         graph
#     pos_attr : str
#         name of the node attribute given the position of the nodes 
#     attr : str, optional
#         name of the node attribute to be plotted
#     attr_ind : int, default: 0
#         index of the attribute to be plotted (used if attr is not `None`)
#     attr_label : str, optional
#         label for the colorbar, used if `attr` is not `None`;
#         by default (`None`): `attr_label` is set to `attr`_`attr_ind`
#     with_labels : bool, default: False
#         indicates if the node id is plotted (`True`) or not (`False`)
#     plotter : :class:`pyvista.Plotter`, optional
#         - if given (not `None`), add element to the plotter, a further call to \
#         `plotter.show()` will be required to show the plot
#         - if not given (`None`, default): a plotter is created and the plot \
#         is shown
#     show_scalar_bar : bool, default: True
#         indicates if the scalar bar is displayed (`True`) or not (`False`);
#         used if `attr` is not `None`
    
#     kwargs_edges : dict, optional
#         keyword arguments passed to function `plotter.add_mesh` to plot edges,
#         possible keys, values : 
        
#         - 'line_width':3,  # edge width
#         - 'color':'black', # edge color 
#         - 'opacity':.8,    # opacity
        
#     kwargs_pts : dict, optional
#         keyword arguments passed to function `plotter.add_mesh` to plot nodes (points),
#         possible keys, values :         
        
#         - 'render_points_as_spheres':True,  # rendering
#         - 'point_size':10,                  # point size
#         - 'color':'black',                  # point color (if attr=None)
#         - 'cmap':'plasma',                  # color map (if attr is not None)
#         - 'clim':[vmin, vmax],              # min and max for color map (if attr is not None)
    
#     kwargs_pts_labels : dict, optional
#         keyword arguments passed to function `plotter.add_point_labels` to plot labels
#         (used if `with_labels=True`), possible keys, values : 
        
#         - 'point_size':0, # point size
#         - 'font_size':24, # font size
#         - 'color':'black',
    
#     kwargs_scalar_bar : dict, optional
#         keyword arguments passed to function `plotter.add_scalar_bar` to plot scalar bar
#         (used if `attr` is not None), possible keys, values : 
        
#         - 'vertical':True,
#         - 'title_font_size':24,
#         - 'label_font_size':12,

#     cpos : sequence[sequence[float]], optional
#         camera position (unsused if `plotter=None`);
#         `cpos` = [camera_location, focus_point, viewup_vector], with

#         - camera_location: (tuple of length 3) camera location ("eye")
#         - focus_point    : (tuple of length 3) focus point
#         - viewup_vector  : (tuple of length 3) viewup vector (vector \
#         attached to the "head" and pointed to the "sky")

#         note: in principle, (focus_point - camera_location) is orthogonal to
#         viewup_vector
#     """
#     # fname = 'plot_graph_3d'

#     mesh = get_mesh_from_graph(G)

#     if kwargs_edges is None:
#         kwargs_edges = {}

#     if kwargs_pts is None:
#         kwargs_pts = {}

#     if kwargs_scalar_bar is None:
#         kwargs_scalar_bar = {}

#     if kwargs_pts_labels is None:
#         kwargs_pts_labels = {}

#     if plotter is not None:
#         pp = plotter
#     else:
#         pp = pv.Plotter()

#     pp.add_mesh(mesh, **kwargs_edges)           # draw edges
    
#     if attr is None:
#         pp.add_mesh(mesh.points, **kwargs_pts)  # draw nodes
#     else:
#         # get value of the attribute (and index) to plot        
#         v = np.asarray(list(networkx.get_node_attributes(G, attr).values()))[:, attr_ind]
        
#         if attr_label is None:
#             attr_label = f'{attr}_{attr_ind}'

#         # Set points with color
#         pts = pv.PolyData(mesh.points)
#         pts[attr_label] = v # colors for the points

#         if 'cmap' in kwargs_pts.keys():
#             if 'color' in kwargs_pts.keys():
#                 del kwargs_pts['color']

#         if 'title' not in kwargs_scalar_bar:
#             scalar_bar_title_added = True
#             kwargs_scalar_bar['title'] = attr_label
#         else:
#             scalar_bar_title_added = False

#         pp.add_mesh(pts, **kwargs_pts, show_scalar_bar=False)    # draw nodes
#         if show_scalar_bar:
#             pp.add_scalar_bar(**kwargs_scalar_bar)                   # add scalar bar (color bar)

#         if scalar_bar_title_added:
#             del kwargs_scalar_bar['title']

#     if with_labels:
#         pp.add_point_labels(mesh.points, np.arange(mesh.n_points), **kwargs_pts_labels) # add point labels
    
#     if plotter is None:
#         pp.show(cpos=cpos)

# # -----------------------------------------------------------------------------
# def apply_pca_to_node_features(G, attr='pos', new_attr=None, normalize=False, normalize_eps=1.e-10, return_trans=False):
#     """
#     Apply PCA to node features (specified by attribute name) of a graph.

#     New features are the original features expressed in the PCA system, centered
#     at the mean of the original features; moreoever, normalization (wrt. 
#     standard deviation along principal axes) is done optionally.

#     Parameters
#     ----------
#     G : networkx.Graph object
#         graph, with nodes with integer id starting from 0

#     attr : str, default: 'pos'
#         name of the attribute on which PCA is done; the attribute is assumed 
#         to have at least 2 features

#     new_attr : str, optional
#         name of the transformed attribute;
#         by default (`None`): `new_attr` is set to `attr`, i.e. transformation 
#         is done in-place

#     normalize : bool, default: `False`
#         - if `True`: new features are normalized by the standard deviation along 
#         principal axes
#         - if `False`: new features are not normalized

#     return_trans : bool, default: `False`
#         - if `True`: transformation `(node_features_mean, smat, d)` is returned 
#         (see Returns below)
#         - if `False`: transformation is not returned

#     Returns
#     -------
#     node_features_mean : 1d numpy array, optional
#         mean features (centered of the PCA system)
#     smat : 2d numpy array, optional
#         returned if `return_trans=True`:
#         PCA system, i.e. matrix whose columns are the principal axes (of norm 1)
#     d : 1d numpy array, optional
#         returned if `return_trans=True`:
#         standard deviation along prinicpal axes
#     """

#     # Node features as 2d array with 
#     node_features_array = np.array(list(networkx.get_node_attributes(G, attr).values()))

#     # Covariance matrix of node features
#     cov_mat = np.cov(node_features_array.T)

#     # Diagonalization
#     d, smat = np.linalg.eig(cov_mat)
#     # -> d: (1d-array) eigen values, variances along prinicpal axes
#     # -> smat: matrix whose columns are the eigen vectors (of norm 1), i.e. principal axes
#     d = np.sqrt(d) # std along principal axes

#     # sort variances in descending order
#     ind = np.argsort(d)[::-1]
#     smat = smat[:,ind]
#     d = d[ind]
    
#     # Apply rotation wrt. PCA
#     node_features_mean = node_features_array.mean(axis=0)
#     node_features_array = (node_features_array-node_features_mean).dot(smat)

#     if normalize:
#         # Normalize according to std
#         node_features_array = node_features_array/(d+normalize_eps)
   
#     if new_attr is None:
#         new_attr = attr
#     networkx.set_node_attributes(G, {i: nf for i, nf in enumerate(node_features_array)}, new_attr)

#     if return_trans:
#         out = node_features_mean, smat, d
#     else: 
#         out = None

#     return out
# # -----------------------------------------------------------------------------

# # ----------------------------------------------------------------------------
# def rotationMatrix2D(alpha=0.0):
#     """
#     Returns a 2x2 matrix of rotation in 2D.

#     The function returns the 2x2 matrix m for changing the coordinates system
#     from Ox'y' to Oxy, where the system Ox'y' is obtained from the system Oxy by
#     applying a rotation of angle -alpha,
#             +                         +
#             |  cos(alpha)   sin(alpha)|
#         m = | -sin(alpha)   cos(alpha)|
#             +                         +
#     i.e. the column of m are the new axes expressed in the system Oxy.

#     Parameters
#     ----------
#     alpha : float, default: 0.0
#         azimuth angle in degrees

#     Returns
#     -------
#     mrot : 2D array of shape (2, 2)
#         rotation matrix (see above)
#     """
#     a = alpha * np.pi/180.0
#     ca, sa = np.cos(a), np.sin(a)
#     m = np.array([[ca, sa], [-sa, ca]])
#     return m
# # ----------------------------------------------------------------------------

# # ----------------------------------------------------------------------------
# def rotationMatrix3D(alpha=0.0, beta=0.0, gamma=0.0):
#     """
#     Returns a 3x3 matrix of rotation in 3D.

#     The function returns the 3x3 matrix m for changing the coordinates system
#     from Ox'''y'''z''' to Oxyz, where the system Ox'''y''''z''' is obtained
#     from the system Oxyz as follows:
#         Oxyz      -- rotation of angle -alpha around Oz  --> Ox'y'z'
#         Ox'y'z'   -- rotation of angle -beta  around Ox' --> Ox''y''z''
#         Ox''y''z''-- rotation of angle -gamma around Oy''--> Ox'''y'''z''',
#             +                                                            +
#             |  ca * cc + sa * sb * sc,  sa * cb, - ca * sc + sa * sb * cc|
#         m = |- sa * cc + ca * sb * sc,  ca * cb,   sa * sc + ca * sb * cc|
#             |                 cb * sc,     - sb,                  cb * cc|
#             +                                                            +
#     where
#         ca = cos(alpha), cb = cos(beta), cc = cos(gamma),
#         sa = sin(alpha), sb = sin(beta), sc = sin(gamma)
#     i.e. the column of m are the new axes expressed in the system Oxyz.

#     Parameters
#     ----------
#     alpha : float, default: 0.0
#         azimuth angle in degrees
#     beta : float, default: 0.0
#         dip angle in degrees
#     plunge : float, default: 0.0
#         azimuth angle in degrees

#     Returns
#     -------
#     mrot : 2D array of shape (3, 3)
#         rotation matrix (see above)
#     """
#     t = np.pi/180.0
#     a = alpha * t
#     b = beta * t
#     c = gamma * t
#     ca, sa = np.cos(a), np.sin(a)
#     cb, sb = np.cos(b), np.sin(b)
#     cc, sc = np.cos(c), np.sin(c)
#     m = np.array(
#             [[  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc],
#              [- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc],
#              [                 cb * sc,     - sb,                   cb * cc]])
#     return m
# # ----------------------------------------------------------------------------
