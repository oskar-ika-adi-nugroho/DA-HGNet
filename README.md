# DA-HTCN: Differential-Attention Hyperedge Temporal Convolution Network

This code is based on Hyperformer  [Hypergraph Transformer for Skeleton-based Action Recognition.](https://arxiv.org/pdf/2211.09590.pdf). We made modifications for DA-HTCN.
This repository contains the reference implementation of **DA-HTCN**, a hybrid skeleton-based action recognition model that combines:

- **Differential Hyperedge Self-Attention (DHSA)** for topology-aware global spatial interaction
- **Multi-Scale Temporal Convolution (MultiScale-TCN)** for efficient temporal modeling
- An **early-layer graph branch (Layers 1–4)** to strengthen local spatial grounding

## Figures

### Figure 1. Overall framework
![Figure 1](DA-HTCN_README_assets/Figure1.png)

### Figure 2. DA-HTCN block and DHSA components
![Figure 2](DA-HTCN_README_assets/Figure2.png)

### Figure 3. Two-branch differential attention in DHSA
![Figure 3](DA-HTCN_README_assets/Figure3.png)

### Figure 4. Enriched five-stream (RICH5) decomposition for skeleton-based action recognition
![Figure 4](DA-HTCN_README_assets/Figure4.png)

## Main idea (high level)

Given an input sequence of 3D skeleton joints \(X \in \mathbb{R}^{N\times C\times T\times V\times M}\), DA-HTCN:
1. Normalizes and reshapes the input so each person instance is processed as an independent sample in the backbone (\(B=N\cdot M\)).
2. Applies **DHSA** to compute structure-aware attention with:
   - hop-distance RPE from the physical skeleton graph
   - hyperedge context tokens from joint-to-part pooling
   - differential attention (two attention branches combined by subtraction) to suppress shared noisy correlations
3. Uses **MultiScale-TCN** to aggregate motion patterns over time with multiple dilated temporal branches.
4. Pools features globally and averages over persons for classification.

## Modifications / Changes from upstream (Hyperformer)

DA-HTCN is derived from the Hyperformer codebase, but differs in several key aspects:

1. **Differential attention in the spatial self-attention**
   - Replace the single attention map with a **two-branch attention** design.
   - The final attention response is formed by subtracting the second branch from the first using a learnable coefficient (noise-canceling behavior).

2. **Differential Hyperedge Self-Attention (DHSA) formulation**
   - Keep **hop-based RPE** and **hyperedge-token interactions** (joint-to-part pooling), but integrate them into both attention branches consistently.
   - Use a unified DHSA description aligned with Section 3 of the manuscript (Figures 2–3).

3. **Early-layer graph branch (Layers 1–4 only)**
   - Add an explicit GCN branch only in the first four blocks for stronger kinematic inductive bias early in the network.
   - Two variants are supported:
     - **DA-HTCN-MGCN**: masked (edge-importance) topology over a fixed K-partition adjacency
     - **DA-HTCN-MA-GCN**: masked topology + **sample-adaptive adjacency** blended with a learnable scalar gate

4. **Hybrid spatial fusion policy**
   - In Layers 1–4, DHSA and GCN outputs are fused by **summation** inside the spatial sub-layer before DropPath and residual addition.
   - In Layers 5–10, the GCN branch is removed and the spatial module reduces to DHSA only.

5. **RICH5 input decomposition support (J/E/S/D/A)**
   - In addition to classic 4-stream settings, DA-HTCN supports the enriched five-stream decomposition:
     - **J**: joint coordinates (root-normalized)
     - **E**: edge/bone vectors
     - **S**: surface (cross-product) features
     - **D**: motion (temporal difference)
     - **A**: acceleration-like (second temporal difference)
   - Late fusion is implemented as a weighted sum of per-stream class probabilities.

6. **Engineering updates for stability and reproducibility**
   - Minor refactoring and device-safe buffers (e.g., hop-distance tensor handling) to improve training stability and reproducibility across environments.

## Results on NTU RGB+D 60

Table 1 below is reproduced from the draft manuscript (X-Sub and X-View):

| Architecture | Method | X-Sub (%) | X-View (%) |
| --- | --- | --- | --- |
| Graph Convolution | ST-GCN [1] | 81.5 | 88.3 |
|  | 2S-AGCN [11] | 88.5 | 95.1 |
|  | Shift-GCN [12] | 90.7 | 96.5 |
|  | SGN [26] | 89.0 | 94.5 |
|  | CTR-GCN [27] | 92.4 | 96.8 |
|  | Info-GCN [28] | 93.0 | 97.1 |
|  | HLP-GCN [29] | 92.7 | 96.9 |
|  | HD-GCN [30] | 93.4 | 97.2 |
| Transformer | ST-TR [31] | 89.9 | 96.1 |
|  | DSTA [32] | 91.5 | 96.4 |
|  | Hyperformer [9]* | 92.9 | 95.1 |
| Hybrid Model(GCN + Att) | Dynamic GCN [33] | 91.5 | 96.0 |
|  | EfficientGCN-B4 [34] | 91.7 | 95.7 |
|  | DA-HTCN (ours) | 93.48 | 96.70 |

\* The Hyperformer entry is listed as in the manuscript table.

## Multi-stream setting (RICH5)

DA-HTCN supports both:
- Standard 4-stream: Joint, Bone, Joint Motion, Bone Motion
- Enriched 5-stream (RICH5): **J/E/S/D/A** (Joint, Edge, Surface, Motion, Acceleration)

Each stream can be trained with either DA-HTCN-MGCN or DA-HTCN-MA-GCN, and combined by late fusion (weighted sum of per-stream class probabilities).

## Training (NTU RGB+D 60, typical)

- Framework: PyTorch
- Epochs: 140
- Loss: cross-entropy
- Base LR: 0.025, decayed by 0.1 at epochs 110 and 120
- Batch size: 32
- Temporal length: sequences resized to 64 frames

(These settings match the configuration used in the accompanying manuscript and config files.)

## Citation

If you use DA-HTCN in your research, please cite the associated paper (manuscript) and the relevant baselines used in your experiments.
