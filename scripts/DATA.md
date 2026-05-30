# Data Processing

## Degradation Pipelines

### 1. Degradation Pipeline I

```math
I_{LR} = \mathcal{D}(I_{HR})
```
- $I_{HR}$ - HR image
- $I_{LR}$ - LR image
- $\mathcal{D}$ - downsampling operator

### 2. Degradation Pipeline II

```math
I_{LR} = \mathcal{D}(\mathcal{B}(I_{HR})) + {\mathcal{N}}
```
- $\mathcal{D}$ - downsampling operator
- $\mathcal{B}$ - blurring operator
- $\mathcal{N}$ - noise term