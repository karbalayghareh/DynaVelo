# DynaVelo
A latent neural ODE model to learn multiomic velocities of cells, infer dynamic gene regulations, and do in-silico perturbations to predict velocity changes.

### Environment Setup

1. Make sure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
2. Clone this repository:

```bash
git clone https://github.com/karbalayghareh/DynaVelo.git
cd DynaVelo

conda env create -f environment.yml
conda activate dynavelo
```

### Usage
Two adata files are required: adata_rna and adata_atac. The first adata contains preprocessed RNA expression values and RNA velocity estimates from scVelo. As RNA velocities are prone to error and sensitive to gene sets, we have to first check if the overall trajectories make biological sense. The second adata contains TF motif accessibility z-scores from chromVAR. The shape of adata_rna is [n_cells, n_genes], and the shape of adata_atac is [n_cells, n_tfs].

#### Training a DynaVelo model
We first build Pytorch datasets and dataloaders from adata_rna and adata_atac to be used for training. MultiomeDataset is used to create the custom datasets. We randomly allocate 10% of the cells for the test and the rest for the training dataset.

```python
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dynavelo.models import MultiomeDataset
from dynavelo.models import DynaVelo

# set seed
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# set gpu
gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')

# datasets and dataloaders
dataset = MultiomeDataset(adata_rna, adata_atac)
N_test = int(0.1 * len(dataset))
idx_random = np.random.permutation(len(dataset))
idx_test = idx_random[:N_test]
idx_train = idx_random[N_test:]
dataset_train = MultiomeDataset(adata_rna[idx_train], adata_atac[idx_train])
dataset_test = MultiomeDataset(adata_rna[idx_test], adata_atac[idx_test])

dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=0)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

# model
model = DynaVelo(x_dim=adata_rna.shape[1], y_dim=adata_atac.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# train
model.fit(dataloader_train, dataloader_test, optimizer, max_epoch=200)
```

#### Predicting velocities and latent times
After a DynaVelo model is trained, we use it to predict the latent times, RNA velocities, and motif velocities for all cells, saving the results in the adata files. The mode evaluation-sample means that we sample from the learned posterior probabilities of initial points in the latent space and latent times of cells `n_samples` times, then report the mean and variance of the predicted velocities. The variance can represent uncertainty in the velocities.

```python
model.mode = 'evaluation-sample'
adata_rna_pred, adata_atac_pred = model.evaluate(adata_rna, adata_atac, dataloader, n_samples=20)
```

#### Calculate Jacobian matrices
To learn dynamic and cell-state-specific gene regulatory networks (GRNs), we calculate the Jacobian matrices of the trained DynaVelo model. There are four types of Jacobian matrices:

(1) J_vx_x, which measures the partial effects of RNA expression on RNA velocity and has the shape [n_cells, n_genes, n_genes].<br> 
(2) J_vy_x, which measures the partial effects of RNA expression on motif velocity and has the shape [n_cells, n_tfs, n_genes].<br> 
(3) J_vx_y, which measures the partial effects of TF motif accessibility on RNA velocity and has the shape [n_cells, n_genes, n_tfs].<br> 
(4) J_vy_y, which measures the partial effects of TF motif accessibility on motif velocity and has the shape [n_cells, n_tfs, n_tfs].

Since the Jacobians are 3D dense tensors, they take a lot of memory, so we choose a subset of genes we are interested in and calculate the Jacobians for them. We include all the TFs in the subset of `genes_of_interest`, as we are interested in understanding how TFs regulate each other. The mode evaluation-fixed means that we use the mean of the latent times and initial points of the cells in the latent space without sampling.

```python
# Jacobians
model.mode = 'evaluation-fixed'
adata_rna_pred = model.calculate_jacobians(model, adata_rna_pred, adata_atac_pred, dataloader, genes_of_interest, epsilon = 1e-4)
```

#### In-silico gene perturbation
One of the useful applications of DynaVelo is to perform in-silico gene perturbations and observe the resulting changes in RNA and motif velocities. This approach helps us understand how perturbing a gene can alter cell trajectories. Such insights are invaluable for identifying optimal perturbation targets to restore lost functions in diseases where normal trajectories have been disrupted. The `perturbed_genes` list specifies the genes for in-silico perturbations. If a gene is a TF, both its RNA expression and motif accessibility are set to the minimum value observed across all cells; otherwise, only RNA expression is perturbed.

```python
# In-silico gene perturbation
model.mode = 'evaluation-fixed'
adata_rna_pred = model.predict_perturbation(model, adata_rna_pred, adata_atac_pred, dataloader, perturbed_genes)
```





