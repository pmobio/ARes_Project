# LLM-powered construction of antibiotic resistance datasets and knowledge graphs for accelerated antibiotic resistance gene discovery

<img src="figure/Figure_abstract.png" width="1000">



## 📌 News 📌

**Current release (v1.0, 2025).**

 **AResDB**: 22,414 experimental samples spanning 2,769 microbial strains, 3,379 compounds, 359 resistance-associated genes, 468 substitution mutations, and 352 distinct resistance mechanisms.

 **AResKG**: 51,704,105 triples across 14 biological entity types and 27 relation types, providing a unified, multi-species representation of antibiotic resistance knowledge.



## Contents

**Benchmarking across top 5 species.** We evaluate five representative models—PRA, MLP, Stacked (PRA+MLP), GCN, and GAT—on the five species with the largest curated datasets in AResDB: *Staphylococcus aureus*, *Escherichia coli*, *Pseudomonas aeruginosa*, *Mycobacterium tuberculosis*, and *Klebsiella pneumoniae*.

**AResKG vs. existing graphs.** We compare models trained/evaluated on AResKG against KIDS to quantify the effects of data scale, structural completeness, and semantic consistency on predictive performance.



## Repository

- **Data:** contains  source data of  AResDB and AResKG, and top 5 species 
- **Demo:** contains the source code for model training, evaluation, and performance visualization.
- **Results:** contains prediction outputs and evaluation metrics.
- **Figure:** contains generated figures for two benchmarking scenarios — (i) multi-species benchmarking and (ii) graph comparison between **AResKG** and **KIDS**.



## Getting Started

```
# Step 1: Create environment
$ conda env create -f environment.yaml

# Step 2: Activate environment
$ conda activate areskg
```



## Running the Code

This repository supports both **statistical** and **graph-based** models for antibiotic resistance link prediction.



1、**PRA / MLP / Stacked Models**

1、PRA**, **MLP**, and **Stacked (PRA + MLP)** models with **k-fold cross-validation** for antibiotic resistance link prediction. Refer to the KIDS [README.md](https://github.com/IBPA/KIDS/blob/master/hypothesis_generator/README.md) for detailed architecture and parameter definitions.

```
cd run
./run_pra_experiments.sh
./run_er_mlp_experiments.sh
./run_stacked_experiments.sh
```

Each script performs 5-fold training and save metrics (Precision, Recall, F1-score, and AUCPR) to `/results/`.



2、**GNN Models** 

Implements **graph neural networks** using **PyTorch Geometric** to learn structured embeddings from AResKG.

```
$ cd /gnn 
$ python train_gnn.py --fold 0 --method GCN --strain Klebsiella  --nega_num 3 --lr 1e-4
$ python train_gnn.py --fold 0 --method GAT --strain Klebsiella  --nega_num 6 --lr 1e-4
```

*Key Arguments*

| Parameter           | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `--strain`          | Bacterial strain name (*Klebsiella*, *E.coli*, etc.) |
| `--fold`            | Cross-validation fold (0–4)                          |
| `--method`          | GNN method (`GCN` or `GAT`)                          |
| `--lr`              | Learning rate (default = 1e-2)                       |
| `--in_channels`     | Input dimension (default 256)                        |
| `--hidden_channels` | Hidden layer dimension (default = 256)               |
| `--out_channels`    | Output dimension (default 256)                       |
| `--nega_num`        | Negative sample size                                 |

*Model Architechture*

- **GCN** – Graph Convolutional Network

- **GAT** – Graph Attention Network

  Each model uses an **MLP decoder** for link prediction between entities.

  

## Example output

<img src="figure/Figure1_AResKG_top5_species_boxplot.png" width="1000">

Figure 1. Comparative performance of five representative methods (PRA, MLP, Stacked, GCN, and GAT) on the top five bacterial species in our dataset.


<img src="figure/Figure2_Box_plot_compare.png" width="1000">

Figure 2. Models trained and evaluated on the AResKG knowledge graph achieve consistently higher predictive accuracy than those trained on existing antibiotic-resistance graphs, i.e., KIDS.

## Coypright

AResDB and AResKG were created and maintained by the Tencent AI for Life Sciences Lab.  

© Tencent AI for Life Sciences Lab. All rights reserved.  

Unauthorized reproduction, redistribution, or commercial use is prohibited without prior permission.
