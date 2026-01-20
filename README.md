# fracture_network_deepgen
Example of fracture networks generation using GraphRNN and DDPM.

<br>
<img src="./anim/06_00_anim_2d.gif" alt="ddpm">
<br/>


This repository is associated with forthcoming abstract and paper:

**Fracture network modeling with graph deep learning**

**Natural fracture network generation using graph deep learning**


Ana Paula Burgoa Tanaka, Philippe Renard, Julien Straubhaar, Xiao Xia Liang, Dany Lauzon 


## Journal reference
Forthcoming

## Data description
Reference fracture network interpretation data from: 

Graph-based fracture network analysis to integrate structural geology properties and identify preferential flow pathways in the aquifer system of Tsanfleuron, Swiss Alps. Journal of Structural Geology. Volume 201.

https://github.com/anapaulabtanaka/tsanfleuron_fracture_networks


https://doi.org/10.5281/zenodo.15739431


Training dataset from various open datasets of fracture interpretation:

Aland Island (Finland): https://doi.org/10.5281/zenodo.4719627


Apodi (Brazil): https://doi.org/10.4121/uuid:988152da-3ac3-44cb-9d87-c7365e3707b6


Australia: https://doi.org/10.26180/5cdcad0a73fe0


Brejoes (Brazil): https://doi.org/10.4121/uuid:67cde05c-9e99-4cc4-8cec-9f2666457d1f


Bristol Channel (UK): https://doi.org/10.4121/14039234.v1


Bristol Patterns (UK): https://doi.org/10.5194/se-11-1773-2020


Coastal (Ireland): https://www.data.gov.uk/dataset/c8cc3d41-f5ec-4661-9d9b-75bff515b3c5/fault-or-fracture-trace4


Forsmark (Sweden): https://zenodo.org/records/4250778


Mineral Mountains (USA): https://doi.org/10.15121/1405031


Outcrop and syntehtic patterns: Generated based in https://doi.org/10.1002/2017WR020943


Synthetic offshore (Brazil): Generated based in https://doi.org/10.1016/j.marpetgeo.2021.105392


Tsanfleuron (Switzerland): Same as reference fracture network

Parmelan (France): https://doi.org/10.4121/uuid:3f5e255f-edf7-441f-89f2-1adc7ac2f7d1


Salta (Argentina): Generated based in 



### Files
In the "data" folder, you will find:
 
- Reference fracture network graph for the generation of new netwroks with GraphRNN and DDPM - Tsanfleuron
  - Filename: `tsan_largest_cc.pickle`
  - Filename: `tsansimple_largest_cc.pickle`
 
- Fracture networks as graphs for model learning - From fracture network open datasets and build synthetic networks
  - Synthetic patterns: `brick.pickle`, `diamond.pickle`, `hexagon.pickle`, `pavement.pickle`, `polygonal.pickle`, `star.pickle`, `stochastic.pickle`
  - Outcrop patterns: `out_braided.pickle`, `out_brick.pickle`, `out_diamond.pickle`, `out_hexagon.pickle`, `out_pavement.pickle`, `out_polygonal.pickle`, `out_star.pickle`, `out_stochastic.pickle`
  - Aland Island (Finland): `.pickle`
  - Apodi (Brazil): `.pickle`
  - Australia: 
  - Brejoes (Brazil): `.pickle`
  - Bristol Channel (UK): `pickle`
  - Vristol Patterns (UK):
  - Coastal (Ireland): `.pickle`
  - Forsmark (Sweden): `.pickle`
  - Mineral Mountains (USA): `.pickle`
  - Synthetic offshore (Brazil): `.pickle`
  - Tsanfleuron (Switzerland): `.pickle`

  - Parmelan (Frace): `.pickle` -to add-verify geometry-
  - Salta (Argentina): `.pickle` -to add-verify geometry-
 
- Pre-processed graphs for training derived from the above networks
  - Filename: `G0.pickle`, `G1.pickle`, ..., `G176.pickle`

## Dependencies
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages

```bash
pip install matplotlib, numpy, scipy, networkx, pyvista, karstnet, pytorch, torch_geometric, cuda, shapely, geopandas
```

## Generation based on the method proposed for karst generation

For the GraphRNN and DDPM please check: [https://github.com/ERC-Karst/karst_networks_gen](https://github.com/ERC-Karst/karst_networks_gen_public/tree/v1.0.0)


In: Lauzon, D., Straubhaar, J., Renard, P. A deep generative model for the simulation of discrete karst networks. Earth and Space Science, 12. https://doi.org/10.1029/2025EA004360


In: Julien Straubhaar. (2025). ERC-Karst/karst_networks_gen_public: Version 1.0.0, submitted to Earth and Space Science (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.15090730


## Examples

For transformation of the fracture network interpretation in graphs, definition of most connected components, set identification, length and azimuth calculation, storage and export:

`Generate graph and export largest connected component` : generate graph from interpretation, defining most connected components and exporting subgraphs.

`Set definition, length and azimuth calculation and storage` : set identification, length and azimuth calculation and storage as attributes.


For the generation of fracture networks the examples are grouped in the folders "gen_{scenario_name}". The examples are from the karst_networks_gen repository (reference above).

`00_graphData_collection.ipynb` : generate a collection of subgraphs (from the main graph) for data set and test set.

`01_graphRNN_model_train.ipynb` : define and train the GraphRNN model.

`02_graphRNN_model_play.ipynb` (optional step) : play / test the Graph RNN model for graph generation (topology only).

`03_graphDDPM_model_train.ipynb` : define and train the GraphDDPM model for node features generation.

`04_graphDDPM_model_play.ipynb` (optional step) : play / test the Graph DDPM model for node features generation.

`05_gen_graph.ipynb` : generate an ensemble of graphs (topology + node features) : the topology is generated using the Graph RNN model, then the node features are generated using Graph DDPM model.

`06_gen_graph_anim.ipynb` (optional step) : animation of the denoising process.

`07_gen_graph_stats.ipynb` : compute statistics on generated graphs and on the graphs from the data set.


