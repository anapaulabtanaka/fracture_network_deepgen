# fracture_network_deepgen
Example of fracture networks deep generation using GraphRNN and DDPM.

<br>
<img src="./anim/06_00_anim_2d.gif" alt="ddpm">
<br/>


This repository is associated with forthcoming paper:

**Fracture network generation using graph deep learning**

Ana Paula Burgoa Tanaka <sup>*</sup>, Philippe Renard, Julien Straubhaar, Xiao Xia Liang 


## Journal reference
Forthcoming

## Data description
Open fracture networks interpretation data from: 

Graph-based fracture network analysis to integrate structural geology properties and identify preferential flow pathways in the aquifer system of Tsanfleuron, Swiss Alps. Journal of Structural Geology. Volume 201.

https://github.com/anapaulabtanaka/tsanfleuron_fracture_networks


https://doi.org/10.5281/zenodo.15739431

### Files
In the "data" folder, you will find:
 
- Links and nodes for the graph generation with GraphRNN and DDPM
  - Filename: `Tsanfleuron_nodes.dat`
  - Filename: `Tsanfleuron_links.dat`

## Deep graph generation

For the GraphRNN and DDPM please check: [https://github.com/ERC-Karst/karst_networks_gen](https://github.com/ERC-Karst/karst_networks_gen_public/tree/v1.0.0)
10.5281/zenodo.15090730

In: Lauzon, D., Straubhaar, J., Renard, P. A deep generative model for the simulation of discrete karst networks. Earth and Space Science, 12. https://doi.org/10.1029/2025EA004360



## Dependencies
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages.

```bash
pip install matplotlib, numpy, scipy, networkx, pyvista, karstnet, pytorch, torch_geometric, cuda
```
## Examples

Some notebooks examples:
