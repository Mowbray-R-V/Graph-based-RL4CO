# Tutorial (under progress)
1. [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
2. [Geometric deep learning book](https://geometricdeeplearning.com/book/)
3. [Math behind GNN ](https://rish-16.github.io/posts/gnn-math/)
4. [GCN video](https://www.youtube.com/watch?v=5SintlY9hbY&t=743s)
5. [Beyond Euclid: An Illustrated Guide to Modern Machine Learning with Geometric, Topological, and Algebraic Structures](https://arxiv.org/pdf/2407.09468)
6. [Mathematical Foundations of Geometric Deep Learning](https://arxiv.org/pdf/2508.02723#page=65.06)
7. [Two-Dimensional Weisfeiler-Lehman Graph Neural Networks for Link Prediction](https://arxiv.org/pdf/1709.05584)
8. [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/pdf/1806.01261)
9. [Introduction to Graph Neural Networks: A Starting Point for Machine Learning Engineers](https://arxiv.org/pdf/2412.19419)
10. [awesome-self-supervised-gnn](https://github.com/ChandlerBang/awesome-self-supervised-gnn)
11. [GCN code](https://github.com/tkipf/gcn) **(need to redo from scratch)**

# Types of learning
1. Supervised: Explicit labels for nodes or graphs (e.g., node classification); common loss: Cross-entropy between predicted and true labels
2. Self-supervised/Un-Supervised: No labels — model builds its own pseudo-task (e.g., contrastive learning, context prediction); common loss: InfoNCE, MI maximization
3. Semi-supervised: Labeled + unlabeled data combined; comon loss: Labeled loss + self-supervised regularizer

# Key papers
1. Theory of Graph Neural Networks: Representation and Learning
2. A Comprehensive Survey on Graph Neural Networks, 2019

# Expressiveness
1.B. Weisfeiler and A. Leman, “A reduction of a graph to a canonical form and an algebra arising during this reduction,” Nauchno-Technicheskaya Informatsia, 1968.      
2. How powerful are graph neural network?, ICLR 2019 (many GNNs (Message Passing Neural Networks) are isomorphism-invariant; they cannot distinguish two isomorphic graphs)      
3. go neural: higher-order graph neural networks, AAAI 2019      
4. Graph Positional and Structural Encoder, ICML 2024      
5. Graph Neural Netwroks With Learnable Structural and Positional Representations, ICLR 2022      
6. Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning, NIPS 2020      
7. Demystifying Higher-Order Graph Neural Networks      
8. A Survey on The Expressive Power of Graph Neural Networks      


# Graph neural netwrok
## Spectral methods (Defines convolution in the graph Fourier (spectral) domain using the graph Laplacian’s eigenbasis)
1. Spectral Convolutional Neural Network (Spectral CNN), ICLR 2014  
2. Chebyshev Graph Convolutional Network (ChebNet), NeurIPS 2016  
3. Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017  **(Personal intitution: have a shared weights CNN visualisation in mind to have a better understanding)** GCNs already perform an aggregation that enforces neighbor similarity as an inductive bias.
5. CayleyNet (CayleyNet), ICML 2018  
6. GraphWave / DiffusionWavelet (GraphWave), KDD 2018  
7. Lanczos Network (LanczosNet), ICLR 2019  
8. Spectral Attention Network (SAN), NeurIPS 2020  
9. Fourier Graph Neural Network (Fourier GNN), ICLR 2021  
10. Auto-Regressive Moving-Average Graph Neural Network (ARMA GNN), IEEE TNNLS 2021


## Spatial methods (Defines convolution directly in the node (spatial) domain by aggregating neighbors’ messages) 
These methods basically have two steps 1) Feature smoothing → captures graph structure, 2) Feature transformation → captures feature semantics (learns embeedings in a better reprsentation space speific to the downstream problem)
---------------------------------------
1. GraphSAGE (GraphSAGE), NeurIPS 2017  
2. Graph Attention Network (GAT), ICLR 2018  
3. Message Passing Neural Network (MPNN), ICML 2017  
4. Graph Isomorphism Network (GIN), ICLR 2019  
5. MoNet (Mixture Model Network), CVPR 2017  
6. Edge Convolutional Neural Network (EdgeConv / DGCNN), NeurIPS 2018  
7. Graph Sample and Aggregate (FastGCN), ICLR 2018  
8. GraphSAINT (GraphSAINT), ICLR 2020  
9. Cluster-GCN (Cluster-GCN), KDD 2019
10. Adaptive Graph Convolutional Network (AGCN), AAAI 2018
11. Graph Transformer Network (GTN), NeurIPS 2019  
12. Graph Convolutional Network for Semi-Supervised Learning (GCNII), ICML 2020  
13. Graph Field Network (GFN), NeurIPS 2021


# Oversmoothing 
1. Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning, AAAI 2018
2. Evaluating Deep Graph Neural Networks, 2017
3. NAFS: ASimple yet Tough-to-beat Baseline for Graph Representation Learning, icml 2022
4. OnProvable Benefits of Depth in Training Graph Convolutional Networks, NIPS 2021



# Graph embedding (Converts graphs-high dimensional,non-euclidean space into low-dimensional continuous vector spaces)
1. Understanding Graph Embedding Methods and Their Applications
2. Toward Understanding and Evaluating Structural Node Embeddings, ACM 2021

# Graph based RL4CO
1. GCOMB: Learning budget-constrained combinatorial algorithms over billion-sized graphs — NeurIPS 2020  
2. LeNSE: Learning to navigate subgraph embeddings for large-scale combinatorial optimisation — ICML 2022  
3. PIANO: Influence maximization meets deep reinforcement learning — IEEE Transactions on Computational Social Systems, 2023  
4. DeepIM: Deep graph representation learning and optimization for influence maximization — ICML 2023  
5. ToupleGDD: A fine-designed solution of influence maximization by deep reinforcement learning — IEEE Transactions on Computational Social Systems, 2024  
6. Deep graph representation learning for influence maximization with accelerated inference — Neural Networks  
7. Challenges and opportunities in deep reinforcement learning with graph neural networks: A comprehensive review of algorithms and applications — IEEE Transactions on Neural Networks and Learning Systems, 2022  
8. COMBHELPER: A Neural Approach to Reduce Search Space for Graph Combinatorial Problems  
9. Hierarchical DeepPruner: A Novel Framework for Search Space Reduction  
10. DGN: influence maximization based on deep reinforcement learning  
11. Finding Influencers in Complex Networks: An Effective Deep Reinforcement Learning Approach  
12. Solving the Influence Maximization Problem Via a Deep-Reinforcement-Learning-Guided Evolutionary Approach  
13. DeepSN: A Sheaf Neural Framework for Influence Maximization — AAAI 2025  
14. BiGDN: An end-to-end influence maximization framework based on deep reinforcement learning and graph neural networks — Elsevier 2025  
15. IMNE: Maximizing influence through deep learning-based node embedding in social network — Elsevier 2025  
16. GAWF: Influence maximization method based on graph attention weight fusion  
17. A residual graph reinforcement learning for budgeted influence maximization  
18. Reinforcement-Learning Based Covert Social Influence Operations  
19. Diffusion Model Agnostic Social Influence Maximization in Hyperbolic Space  
20. Location Promoting Influence Maximization in Social Networks  
21. Probing the fitness landscape of the influential nodes for the influence maximization problem in social networks  
22. Finding key players in complex networks through deep reinforcement learning  
23. Multiple Agents Reinforcement Learning Based Influence Maximization in Social Network Services

# GNN scalability 

1. GNN acceleration for large scale graphs — Beyond Message Passing: Neural Graph Pattern Machine, ICML 2025  
2. Survey on graph neural network acceleration: An algorithmic perspective —  2022  
3. Acceleration algorithms in GNNs: A survey — IEEE Transactions on Knowledge and Data Engineering, 2025  
4. A Survey on Graph Neural Network Acceleration: A Hardware Perspective  
5. A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware  
6. A comprehensive survey on distributed training of graph neural networks — ACM Computing Surveys  
7. Distributed Graph Neural Network Training: A Survey  
8. Ripple: Scalable incremental GNN inferencing on large streaming graphs — ICDCS 2025  
9. GNNIE: GNN inference engine with load-balancing and graph-specific caching — 2022  
10. LMC: Fast training of GNNs via subgraph-wise sampling with provable convergence — ICLR 2023  
11. Scaling graph-neural-network training with CPU-GPU clusters  
12. PaSca: A Graph Neural Architecture Search System under the Scalable Paradigm  
13. GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks  
14. A Comprehensive Survey of Dynamic Graph Neural Networks: Models, Frameworks, Benchmarks, Experiments and Challenges  
15. Do Transformers Really Perform Bad for Graph Representation?  

<img width="1046" height="381" alt="image" src="https://github.com/user-attachments/assets/371b1811-d51e-4ed2-bc83-43fe2ebf32c0" />

# GNN generalisation
1. How Powerful are Graph Neural Networks?, ICLR 2019 (Connects GNN expressivity and generalization to the Weisfeiler-Lehman test. Shows limits of message passing.)
2. Fully-Inductive Node Classification on Arbitrary Graphs, ICLR 2025


# Node embedding 
## Proximity  (Capture neighborhood co-occurrence / local similarity. DeepWalk, Node2Vec, matrix-factorization (HOPE, GraRep, LINE), adjacency-factorization, graph autoencoders (GAE/VGAE)) 
## Structural role  (Capture role/functional similarity independent of position--Structural role discovery → identify nodes that play equivalent roles under isomorphism (automorphism groups).) 
1. Roles, positions, and social networks — Stephen P. Borgatti, Martin G. Everett, Social Networks, 1992 (foundational theory of structural roles and equivalence in networks)
2. The role concept in social network analysis — Lorrain & White, Journal of Mathematical Sociology, 1971 (classical notion of structural equivalence)
3. RolX: Structural Role Extraction & Mining in Large Graphs — K. Henderson, B. Gallagher, T. Eliassi-Rad et al., KDD, 2012 (RoleX: automatic role discovery using feature-based NMF decomposition)
4. Revisiting Role Discovery in Networks — K. Henderson, B. Gallagher et al., NIPS Workshop, 2013 (extended RoleX with probabilistic role assignment)
5. struc2vec: Learning Node Representations from Structural Identity — Leonardo F.R. Ribeiro, Pedro Saverese, Daniel R. Figueiredo, KDD, 2017 (hierarchical structural similarity graph + random walks)
6. GraphWave: Learning Structural Node Embeddings via Heat Wavelets — C. Donnat, M. Zitnik, D. Hallac, J. Leskovec, KDD, 2018 (spectral approach using heat diffusion kernel)
   

## Positional  (Encode node location / coordinates in graph (global geometry). Laplacian Eigenmaps, Spectral Embeddings, Distance-to-anchors, LapPE / RWSE for transformers)
## Attribute-aware (Combine node features with structure (often inductive). GCN, GAT, GraphSAGE, HetGNN; autoencoders that reconstruct features+adj )
## Diffusion / Global (Capture long-range influence / diffusion dynamics. PPR / Personalized PageRank embeddings, Heat kernel, APPNP, PPRGo, diffusion maps)
## Contrastive / InfoMax (Learn by pulling positive pairs together and pushing negatives apart (or bootstrapping). DGI, GraphCL, GRACE, MVGRL, InfoNCE losses, BGRL (no-negatives)



#  GNN Simplification (Decouples graph propagation from feature transformation)
1. 



# HPC basic codes for this project


# Libraries
1. PSGraph 
2. [AliGraph](https://github.com/alibaba/graph-learn)
3. [PGL: https](//github.com/PaddlePaddle/PGL)
4. [Euler](https://github.com/alibaba/euler)
5. [SGL](https://github.com/PKU-DAIR/SGL)
