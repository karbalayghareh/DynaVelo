# Author: Alireza Karbalayghareh

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torch import distributions
from torchmetrics.functional import pairwise_cosine_similarity
import scipy
from collections import Counter
from functorch import vmap, jacrev, jacfwd
from utils import EarlyStopping, LRScheduler
import os
from utils import Logger

adjoint = True
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    #from torchsde import sdeint_adjoint as sdeint
else:
    from torchdiffeq import odeint
    #from torchsde import sdeint


class MultiomeDataset(Dataset):
    """
    A custom dataset class for multi-omic data handling RNA and ATAC data.

    Parameters
    ----------
    adata_rna: an anndata object containing RNA expression.
    adata_atac: an anndata object containing TF motif accessibility.
    """

    def __init__(self, adata_rna, adata_atac, use_weights=True):

        self.use_weights = use_weights

        # Process RNA data
        self.x_rna = self._to_dense(adata_rna.X)
        self.vx = np.nan_to_num(self._to_dense(adata_rna.layers['velocity']))

        # Process ATAC data
        self.x_atac = self._to_dense(adata_atac.X)

        # Store observations and variables for RNA and ATAC
        self.rna_obs = adata_rna.obs
        self.atac_obs = adata_atac.obs
        self.rna_var = adata_rna.var
        self.atac_var = adata_atac.var

        # Create a mask for velocity genes
        self.velocity_genes_mask = (~np.isnan(adata_rna.layers['velocity'])).astype(np.float32)

        # Calculate weights for cell types based on their frequencies
        if use_weights:
            self.weights_dict = self._calculate_weights(adata_rna.obs['final.celltype'])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.x_rna.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset.
        """
        x = torch.tensor(self.x_rna[idx, :], dtype=torch.float32)
        y = torch.tensor(self.x_atac[idx, :], dtype=torch.float32)
        vx = torch.tensor(self.vx[idx, :] * self.velocity_genes_mask[idx, :], dtype=torch.float32)
        v_mask = torch.tensor(self.velocity_genes_mask[idx, :], dtype=torch.float32)
        if self.use_weights:
            celltype = self.rna_obs['final.celltype'][idx]
            weight = torch.tensor(self.weights_dict[celltype], dtype=torch.float32)
        else:
            weight = torch.tensor(1., dtype=torch.float32)

        return x, y, vx, v_mask, weight

    def _to_dense(self, matrix):
        """
        Converts a sparse matrix to a dense numpy array.
        """
        return matrix.toarray() if scipy.sparse.issparse(matrix) else matrix

    def _calculate_weights(self, celltypes):
        """
        Calculates weights for each cell type to be used in loss functions or balancing.
        """
        values = celltypes.value_counts()
        weights = 1 / values
        normalized_weights = weights / weights.sum()
        return dict(zip(values.index, normalized_weights))


class node_func(nn.Module):
    """
    Neural ODE function (deep) in the latent space.
    It returns the latent velocities.

    Parameters
    ----------
    zx_dim: dimension of RNA latent space
    zy_dim: dimension of motif latent space
    num_layer: number of layers
    """

    def __init__(self, zx_dim, zy_dim, num_layer):
        super().__init__()

        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.num_layer = num_layer
        layers = []
        for l in range(num_layer):
            if l < self.num_layer - 1:
                layers.append(nn.Linear(zx_dim+zy_dim, zx_dim+zy_dim, bias=True))
                layers.append(nn.GELU())
            else:
                layers.append(nn.Linear(zx_dim+zy_dim, zx_dim+zy_dim, bias=True))
        
        self.node_net = nn.Sequential(*layers)

    def forward(self, t, z):
        return self.node_net(z)


class node_func_wide(nn.Module):
    """
    Neural ODE function (wide) in the latent space.
    It returns the latent velocities.

    Parameters
    ----------
    zx_dim: dimension of RNA latent space
    zy_dim: dimension of motif latent space
    num_hidden: number of hidden neurons in the middle layer
    """

    def __init__(self, zx_dim, zy_dim, num_hidden):
        super().__init__()

        self.fzx = nn.Sequential(
                                nn.Linear(zx_dim+zy_dim, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, zx_dim),
                                )
        
        self.fzy = nn.Sequential(
                                nn.Linear(zx_dim+zy_dim, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, zy_dim),
                                )

        for m in self.fzx.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)

        for m in self.fzy.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)

    def forward(self, t, z):

        fzx = self.fzx(z)
        fzy = self.fzy(z)
        return torch.cat((fzx, fzy), dim=-1)


class node_func_wide_multi_lineage(nn.Module):
    """
    Neural ODE function (wide, multi lineage) in the latent space.
    It returns the latent velocities.

    Parameters
    ----------
    zx_dim: dimension of RNA latent space
    zy_dim: dimension of motif latent space
    h_dim: dimension of augmented space
    num_hidden: number of hidden neurons in the middle layer
    device: GPU device
    """

    def __init__(self, zx_dim, zy_dim, h_dim, num_hidden, device):
        super().__init__()

        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.z_dim = zx_dim + zy_dim
        self.h_dim = h_dim
        self.device = device
        
        self.fzx = nn.Sequential(
                                nn.Linear(zx_dim+zy_dim+h_dim, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, zx_dim)
                                )
        
        self.fzy = nn.Sequential(
                                nn.Linear(zx_dim+zy_dim+h_dim, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, zy_dim)
                                )

        for m in self.fzx.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)

        for m in self.fzy.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)

    def forward(self, t, z):
    
        h = z[..., (self.zx_dim+self.zy_dim):]
        fzx = self.fzx(z)
        fzy = self.fzy(z)

        return torch.cat((fzx, fzy, torch.zeros(h.shape).to(self.device)), dim=-1)


class node_func_wide_time_variant(nn.Module):
    """
    Neural ODE function (wide, time-variant) in the latent space.
    It returns the latent velocities.

    Parameters
    ----------
    zx_dim: dimension of RNA latent space
    zy_dim: dimension of motif latent space
    num_hidden: number of hidden neurons in the middle layer
    """

    def __init__(self, zx_dim, zy_dim, num_hidden):
        super().__init__()

        self.node_net = nn.Sequential(
                                nn.Linear(zx_dim+zy_dim+1, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, zx_dim+zy_dim)
                                )
    def forward(self, t, z):
        if t.dim() == 0:
            t = torch.full_like(z, fill_value=t.item())[:,0:1]
        return self.node_net(torch.cat((t, z), dim=-1))


class DynaVelo(nn.Module):
    """
    DynaVelo model with multiome input.

    Parameters
    ----------
    zx_dim: dimension of RNA latent space
    zy_dim: dimension of motif latent space
    num_hidden: number of hidden neurons in the neural ODE model in the latent space
    k_t: KL divergence coefficient for time
    k_z0: KL divergence coefficient for initial point z0
    k_velocity: coefficient for velocity loss
    k_consistency: coefficient for velocity consistency loss
    mu_pz0: mean of prior normal distribution for z0
    sigma_pz0: standard deviation of prior normal distribution for z0
    alpha_pt: first parameter of prior Beta distribution for t
    beta_pt: second parameter of prior Beta distribution for t
    sigma_x: standard deviation of the normal distribution for reconstructed RNA expression
    sigma_y: standard deviation of the normal distribution for reconstructed motif accessibility
    seed: random seed
    time_variant: if the neural ODE should be time variant
    mode: select the mode (train, evaluation-sample, evaluation-fixed)
    device: select the GPU
    dataset_name: enter name of the dataset
    sample_name: enter name of the sample
    """

    def __init__(self, x_dim, y_dim, 
                 zx_dim=50, zy_dim=50, num_hidden=200, k_t=1000, k_z0=1000, 
                 k_velocity=10000, k_consistency=10000, mu_pz0=0, sigma_pz0=1,
                 alpha_pt=2, beta_pt=2, sigma_x=0.1, sigma_y=0.1, seed=0,
                 time_variant=False, mode='train', device='cuda:0',
                 dataset_name='GCB', sample_name='CtcfWT29'):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.num_hidden = num_hidden
        self.k_t = k_t
        self.k_z0 = k_z0
        self.k_velocity = k_velocity
        self.k_consistency = k_consistency
        self.seed = seed
        self.time_variant = time_variant
        self.mode = mode
        self.device = device
        self.dataset_name = dataset_name
        self.sample_name = sample_name
        self.logger = Logger(["Epoch", "loss_train", "loss_test", "nll_x", "nll_y",  
                              "loss_vel", "loss_con", "kl_z0", "kl_t"], verbose=True)
        
        self.train_dir = f'../checkpoints/{self.dataset_name}/{self.sample_name}/'
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.model_suffix = f'{self.dataset_name}_{self.sample_name}_DynaVelo_num_hidden_{self.num_hidden}_zxdim_{self.zx_dim}_zydim_{self.zy_dim}_k_z0_{str(self.k_z0)}_k_t_{str(self.k_t)}_k_velocity_{str(self.k_velocity)}_k_consistency_{str(self.k_consistency)}_seed_{self.seed}'
        self.ckpt_path = self.train_dir+self.model_suffix+'.pth'

        mu_pz0 = torch.tensor(mu_pz0).to(device)
        sigma_pz0 = torch.tensor(sigma_pz0).to(device)
        self.pz0 = distributions.Normal(loc=mu_pz0, scale=sigma_pz0)

        alpha_pt = torch.tensor(alpha_pt).to(device)
        beta_pt = torch.tensor(beta_pt).to(device)
        self.pt = distributions.Beta(concentration1=alpha_pt, concentration0=beta_pt)

        self.sigma_x = torch.tensor(sigma_x).to(device)
        self.sigma_y = torch.tensor(sigma_y).to(device)

        self.encoder_net_x = nn.Sequential(
                                        nn.Linear(x_dim, zx_dim),
                                        nn.GELU(),
                                        )

        self.encoder_net_y = nn.Sequential(
                                        nn.Linear(y_dim, zy_dim),
                                        nn.GELU(),
                                        )
        
        self.encoder_net_tx = nn.Sequential(
                                        nn.Linear(x_dim, zx_dim),
                                        nn.GELU(),
                                        )

        self.encoder_net_ty = nn.Sequential(
                                        nn.Linear(y_dim, zy_dim),
                                        nn.GELU(),
                                        )
        
        self.decoder_net_x = nn.Sequential(
                                        nn.Linear(zx_dim, x_dim),
                                        nn.ReLU()
                                        )

        self.decoder_net_y = nn.Sequential(
                                        nn.Linear(zy_dim, y_dim),
                                        )

        self.mu_zx0_net = nn.Sequential(
                                        nn.Linear(zx_dim, zx_dim),
                                        )
        self.sigma_zx0_net = nn.Sequential(
                                        nn.Linear(zx_dim, zx_dim),
                                        nn.Softplus()
                                        )

        self.mu_zy0_net = nn.Sequential(
                                        nn.Linear(zy_dim, zy_dim),
                                        )
        self.sigma_zy0_net = nn.Sequential(
                                        nn.Linear(zy_dim, zy_dim),
                                        nn.Softplus()
                                        )

        self.alpha_t_net = nn.Sequential(
                                        nn.Linear(zx_dim+zy_dim, 1),
                                        nn.Softplus()
                                        )

        self.beta_t_net = nn.Sequential(
                                        nn.Linear(zx_dim+zy_dim, 1),
                                        nn.Softplus()
                                        )
        
        if self.time_variant:
            self.node_func = node_func_wide_time_variant(zx_dim, zy_dim, num_hidden)
        else:
            self.node_func = node_func_wide(zx_dim, zy_dim, num_hidden)


    def forward(self, x, y):
        """
        A forward pass through the DynaVelo model.

        Parameters
        ----------
        x: RNA expression matrix with the shape (n_cells_in_batch, n_genes)
        y: TF motif accessibility (chromvar z scores) matrix with the shape (n_cells_in_batch, n_tfs)

        Returns
        -------
        x_pred: predicted RNA expression values
        y_pred: predicted TF motif accessibility values
        vx_pred: predicted RNA velocities for all input genes
        vy_pred: predicted TF motif velocities for all input TFs
        z0_mean: mean of posterior normal distribution for z0
        z0_std: standard deviation of posterior normal distribution for z0
        alpha_t: first parameter of posterior Beta distribution for t
        beta_t: second parameter of posterior Beta distribution for t
        z: latent state
        vz: latent velocity
        latent_time: mean of posterior Beta distribution for t
        latent_time_std: standard deviation of posterior Beta distribution for t
        """

        if self.mode == 'train':
            hx = self.encoder_net_x(x)
            mu_zx0 = self.mu_zx0_net(hx)
            sigma_zx0 = self.sigma_zx0_net(hx)
            zx0 = mu_zx0 + sigma_zx0 * torch.randn(sigma_zx0.shape).to(self.device)

            hy = self.encoder_net_y(y)
            mu_zy0 = self.mu_zy0_net(hy)
            sigma_zy0 = self.sigma_zy0_net(hy)
            zy0 = mu_zy0 + sigma_zy0 * torch.randn(sigma_zy0.shape).to(self.device)

            z0 = torch.cat((zx0, zy0), dim = 1)
            z0_mean = torch.cat((mu_zx0, mu_zy0), dim = 1)
            z0_std = torch.cat((sigma_zx0, sigma_zy0), dim = 1)

            htx = self.encoder_net_tx(x)
            hty = self.encoder_net_ty(y)
            ht = torch.cat((htx, hty), dim = 1)
            alpha_t = self.alpha_t_net(ht)
            beta_t = self.beta_t_net(ht)
            qt = distributions.Beta(concentration1=alpha_t, concentration0=beta_t)
            t_nn = qt.rsample()

            t_nn = t_nn.squeeze()
            t_sorted, indices = torch.sort(t_nn)
            t = torch.cat((torch.tensor([0.]).to(self.device), t_sorted), dim=0)
            t_unique, counts =  torch.unique(t, return_counts=True)
            t_non_unique = t_unique[counts>1]

            while len(t_non_unique)>0:
                    print('Repetitive time')
                    t_nn = qt.rsample()
                    t_nn = t_nn.squeeze()
                    t_sorted, indices = torch.sort(t_nn)
                    t = torch.cat((torch.tensor([0.]).to(self.device), t_sorted), dim=0)
                    t_unique, counts =  torch.unique(t, return_counts=True)
                    t_non_unique = t_unique[counts>1]

            z_all_times = odeint(self.node_func, z0, t,
                            atol=1e-7,
                            rtol=1e-7,
                            method='dopri5',
                        )

            t_reshaped = t.reshape([len(t), 1, 1])
            t_reshaped = t_reshaped.repeat(1, z_all_times.shape[1], 1)
            vz_all_times = self.node_func(t_reshaped, z_all_times)

            z = torch.zeros(z_all_times.shape[1:]).to(self.device)
            vz = torch.zeros(vz_all_times.shape[1:]).to(self.device)
            for idx, cell in enumerate(indices):
                z[cell,:] = z_all_times[idx+1,cell,:]
                vz[cell,:] = vz_all_times[idx+1,cell,:]

            zx = z[:,:self.zx_dim]
            zy = z[:,self.zx_dim:(self.zx_dim+self.zy_dim)]
            vzx = vz[:,:self.zx_dim]
            vzy = vz[:,self.zx_dim:(self.zx_dim+self.zy_dim)]

            x_pred = self.decoder_net_x(zx)
            y_pred = self.decoder_net_y(zy)

            # Jacobians:
            Jx = vmap(jacrev(self.decoder_net_x))(zx)
            Jy = vmap(jacrev(self.decoder_net_y))(zy)

            vx_pred = torch.bmm(Jx, vzx.unsqueeze(2)).squeeze()
            vy_pred = torch.bmm(Jy, vzy.unsqueeze(2)).squeeze()

            return x_pred, y_pred, vx_pred, vy_pred, z0_mean, z0_std, alpha_t, beta_t

        elif self.mode == 'evaluation-fixed':

            hx = self.encoder_net_x(x)
            mu_zx0 = self.mu_zx0_net(hx)
            zx0 = mu_zx0

            hy = self.encoder_net_y(y)
            mu_zy0 = self.mu_zy0_net(hy)
            zy0 = mu_zy0

            z0 = torch.cat((zx0, zy0), dim = 1)
            z0_mean = torch.cat((mu_zx0, mu_zy0), dim = 1)

            htx = self.encoder_net_tx(x)
            hty = self.encoder_net_ty(y)
            ht = torch.cat((htx, hty), dim = 1)
            alpha_t = self.alpha_t_net(ht)
            beta_t = self.beta_t_net(ht)
            t_mean = (alpha_t/(alpha_t+beta_t)).squeeze()

            t_nn = t_mean
            t_sorted, indices = torch.sort(t_nn)
            t = torch.cat((torch.tensor([0.]).to(self.device), t_sorted), dim=0)
            t_unique, counts =  torch.unique(t, return_counts=True)
            t_rep = t_unique[counts>1]
            n_rep = len(t_rep)

            if n_rep==0:
                z_all_times = odeint(self.node_func, z0, t,
                                atol=1e-7,
                                rtol=1e-7,
                                method='dopri5',
                            )

                t_reshaped = t.reshape([len(t), 1, 1])
                t_reshaped = t_reshaped.repeat(1, z_all_times.shape[1], 1)
                vz_all_times = self.node_func(t_reshaped, z_all_times)

                z = torch.zeros(z_all_times.shape[1:]).to(self.device)
                vz = torch.zeros(vz_all_times.shape[1:]).to(self.device)
                for idx, cell in enumerate(indices):
                    z[cell,:] = z_all_times[idx+1,cell,:]
                    vz[cell,:] = vz_all_times[idx+1,cell,:]

            else:
                print(f'n_rep: {n_rep}')
                n_cells = x.shape[0]
                z = torch.zeros(z0.shape).to(self.device)
                vz = torch.zeros(z0.shape).to(self.device)
                for cell in range(n_cells):
                    t = torch.cat((torch.tensor([0.]).to(self.device), t_nn[[cell]]), dim=0)
                    z_cell = odeint(self.node_func, z0[[cell]], t,
                            atol=1e-7,
                            rtol=1e-7,
                            method='dopri5',
                        )
                    t_reshaped = t.reshape([len(t), 1, 1])
                    t_reshaped = t_reshaped.repeat(1, z_cell.shape[1], 1)
                    vz_cell = self.node_func(t_reshaped, z_cell)

                    z[cell,:] = z_cell[1,:,:]
                    vz[cell,:] = vz_cell[1,:,:]

            zx = z[:,:self.zx_dim]
            zy = z[:,self.zx_dim:]
            vzx = vz[:,:self.zx_dim]
            vzy = vz[:,self.zx_dim:]

            x_pred = self.decoder_net_x(zx)
            y_pred = self.decoder_net_y(zy)

            # Jacobians:
            Jx = vmap(jacrev(self.decoder_net_x))(zx)
            Jy = vmap(jacrev(self.decoder_net_y))(zy)

            vx_pred = torch.bmm(Jx, vzx.unsqueeze(2)).squeeze()
            vy_pred = torch.bmm(Jy, vzy.unsqueeze(2)).squeeze()

            latent_time = t_mean
            latent_time_std = torch.sqrt((alpha_t*beta_t)/(((alpha_t+beta_t)**2)*(alpha_t+beta_t+1))).squeeze()
            
            return x_pred, y_pred, vx_pred, vy_pred, z0_mean, z, vz, latent_time, latent_time_std
        
        elif self.mode == 'evaluation-sample':

            hx = self.encoder_net_x(x)
            mu_zx0 = self.mu_zx0_net(hx)
            sigma_zx0 = self.sigma_zx0_net(hx)
            zx0 = mu_zx0 + sigma_zx0 * torch.randn(sigma_zx0.shape).to(self.device)

            hy = self.encoder_net_y(y)
            mu_zy0 = self.mu_zy0_net(hy)
            sigma_zy0 = self.sigma_zy0_net(hy)
            zy0 = mu_zy0 + sigma_zy0 * torch.randn(sigma_zy0.shape).to(self.device)

            z0 = torch.cat((zx0, zy0), dim = 1)
            z0_mean = torch.cat((mu_zx0, mu_zy0), dim = 1)
            z0_std = torch.cat((sigma_zx0, sigma_zy0), dim = 1)

            htx = self.encoder_net_tx(x)
            hty = self.encoder_net_ty(y)
            ht = torch.cat((htx, hty), dim = 1)
            alpha_t = self.alpha_t_net(ht)
            beta_t = self.beta_t_net(ht)
            qt = distributions.Beta(concentration1=alpha_t, concentration0=beta_t)
            t_nn = qt.rsample()
            t_mean = (alpha_t/(alpha_t+beta_t)).squeeze()

            t_nn = t_nn.squeeze()
            t_sorted, indices = torch.sort(t_nn)
            t = torch.cat((torch.tensor([0.]).to(self.device), t_sorted), dim=0)
            t_unique, counts =  torch.unique(t, return_counts=True)
            t_non_unique = t_unique[counts>1]

            while len(t_non_unique)>0:
                    print('Repetitive time')
                    t_nn = qt.rsample()
                    t_nn = t_nn.squeeze()
                    t_sorted, indices = torch.sort(t_nn)
                    t = torch.cat((torch.tensor([0.]).to(self.device), t_sorted), dim=0)
                    t_unique, counts =  torch.unique(t, return_counts=True)
                    t_non_unique = t_unique[counts>1]

            z_all_times = odeint(self.node_func, z0, t,
                            atol=1e-7,
                            rtol=1e-7,
                            method='dopri5',
                        )

            t_reshaped = t.reshape([len(t), 1, 1])
            t_reshaped = t_reshaped.repeat(1, z_all_times.shape[1], 1)
            vz_all_times = self.node_func(t_reshaped, z_all_times)

            z = torch.zeros(z_all_times.shape[1:]).to(self.device)
            vz = torch.zeros(vz_all_times.shape[1:]).to(self.device)
            for idx, cell in enumerate(indices):
                z[cell,:] = z_all_times[idx+1,cell,:]
                vz[cell,:] = vz_all_times[idx+1,cell,:]

            zx = z[:,:self.zx_dim]
            zy = z[:,self.zx_dim:(self.zx_dim+self.zy_dim)]
            vzx = vz[:,:self.zx_dim]
            vzy = vz[:,self.zx_dim:(self.zx_dim+self.zy_dim)]

            x_pred = self.decoder_net_x(zx)
            y_pred = self.decoder_net_y(zy)

            # Jacobians:
            Jx = vmap(jacrev(self.decoder_net_x))(zx)
            Jy = vmap(jacrev(self.decoder_net_y))(zy)

            vx_pred = torch.bmm(Jx, vzx.unsqueeze(2)).squeeze()
            vy_pred = torch.bmm(Jy, vzy.unsqueeze(2)).squeeze()

            latent_time = t_mean
            latent_time_std = torch.sqrt((alpha_t*beta_t)/(((alpha_t+beta_t)**2)*(alpha_t+beta_t+1))).squeeze()
            
            return x_pred, y_pred, vx_pred, vy_pred, z0_mean, z, vz, latent_time, latent_time_std


    def generate_trajectory(self, x, y, t_max, n_points):
        """
        Generates synthetic trajectories using the trained DynaVelo models.

        Parameters
        ----------
        x: a terminal cell's RNA expression with the shape (1, n_genes)
        y: a terminal cell's TF motif accessibility with the shape (1, n_tfs)
        t_max: generate trajectory from time 0 to t_max
        n_points: number of synthetic cells to generate

        Returns
        -------
        x_pred: generated RNA expression with the shape (n_points, n_genes)
        y_pred: generated TF motif accessibility with the shape (n_points, n_tfs)
        vx_pred: generated RNA velocity with the shape (n_points, n_genes)
        vy_pred: generated TF motif velocity with the shape (n_points, n_tfs)
        t_interval: time interval of the generated trajectory (0, t_max)
        """

        hx = self.encoder_net_x(x)
        mu_zx0 = self.mu_zx0_net(hx)
        sigma_zx0 = self.sigma_zx0_net(hx)
        zx0 = mu_zx0 + sigma_zx0 * torch.randn(sigma_zx0.shape).to(self.device)

        hy = self.encoder_net_y(y)
        mu_zy0 = self.mu_zy0_net(hy)
        sigma_zy0 = self.sigma_zy0_net(hy)
        zy0 = mu_zy0 + sigma_zy0 * torch.randn(sigma_zy0.shape).to(self.device)

        z0 = torch.cat((zx0, zy0), dim = 1)
        t_interval = torch.linspace(0, t_max, n_points).to(self.device)
        t = t_interval

        z_all_times = odeint(self.node_func, z0, t,
                        atol=1e-9,
                        rtol=1e-9,
                        method='dopri8',
                    )

        t_reshaped = t.reshape([len(t), 1, 1])
        t_reshaped = t_reshaped.repeat(1, z_all_times.shape[1], 1)
        vz_all_times = self.node_func(t_reshaped, z_all_times)

        z = z_all_times.squeeze()
        vz = vz_all_times.squeeze()

        zx = z[:,:self.zx_dim]
        zy = z[:,self.zx_dim:]
        vzx = vz[:,:self.zx_dim]
        vzy = vz[:,self.zx_dim:]

        x_pred = self.decoder_net_x(zx)
        y_pred = self.decoder_net_y(zy)

        # Jacobians:
        Jx = vmap(jacrev(self.decoder_net_x))(zx)
        Jy = vmap(jacrev(self.decoder_net_y))(zy)

        vx_pred = torch.bmm(Jx, vzx.unsqueeze(2)).squeeze()
        vy_pred = torch.bmm(Jy, vzy.unsqueeze(2)).squeeze()

        return x_pred, y_pred, vx_pred, vy_pred, t_interval


    def train_step(self, dataloader, optimizer):
        """
        Training the model for one epoch.

        Parameters
        ----------
        dataloader: Pytorch dataloader for train data
        optimzer: model optimizer

        Returns
        -------
        loss_all: total loss
        """

        self.train()
        COSLoss = nn.CosineSimilarity(dim=-1, eps=1e-08)
        N = len(dataloader.dataset)
        loss_all = 0

        for idx_batch, (x, y, vx, v_mask, w) in enumerate(dataloader):

            x, y, vx, v_mask, w = x.to(self.device), y.to(self.device), vx.to(self.device), v_mask.to(self.device), w.to(self.device)
            x_pred, y_pred, vx_pred, vy_pred, z0_mean, z0_std, alpha_t, beta_t = self(x, y)

            optimizer.zero_grad()

            #cell weights
            w = w/torch.sum(w)
            w = w.reshape([w.shape[0], 1])

            qz0 = distributions.Normal(loc=z0_mean, scale=z0_std)
            qt = distributions.Beta(concentration1=alpha_t, concentration0=beta_t)
            likelihood_x = distributions.Normal(loc=x_pred, scale=self.sigma_x)
            likelihood_y = distributions.Normal(loc=y_pred, scale=self.sigma_y)
            nll_x = -(w*likelihood_x.log_prob(x)).sum(dim=1).sum(dim=0)
            nll_y = -(w*likelihood_y.log_prob(y)).sum(dim=1).sum(dim=0)
            kl_z0 = distributions.kl_divergence(qz0, self.pz0).sum(dim=1).mean(dim=0)
            kl_t = distributions.kl_divergence(qt, self.pt).sum(dim=1).mean(dim=0)

            vx_pred_masked = vx_pred * v_mask
            vel_loss = -torch.mean(COSLoss(vx_pred_masked, vx))            
            con_loss = torch.mean((pairwise_cosine_similarity(vx_pred) - pairwise_cosine_similarity(vy_pred))**2)

            loss_nll = nll_x + nll_y
            loss_velocity = self.k_velocity * vel_loss 
            loss_velocity_consistency = self.k_consistency * con_loss
            loss_kl = self.k_z0 * kl_z0 + self.k_t * kl_t
            loss = loss_nll + loss_velocity + loss_velocity_consistency + loss_kl

            loss.backward()
            optimizer.step()

            # for logging
            loss_all += loss.detach().cpu().numpy() * x.shape[0]

        return loss_all/N


    def test_step(self, dataloader):
        """
        Testing the model.

        Parameters
        ----------
        dataloader: Pytorch dataloader for test data

        Returns
        -------
        loss_all: total loss
        loss_nll_x_all: negative log-likelihood for x
        loss_nll_y_all: negative log-likelihood for y
        loss_velocity_all: RNA velocity loss
        loss_velocity_consistency_all: velocity consistency loss
        loss_kl_z0_all: KL divergence of z0
        loss_kl_t_all: KL divergence of t
        """

        with torch.no_grad():
            self.eval()
            COSLoss = nn.CosineSimilarity(dim=-1, eps=1e-08)
            N = len(dataloader.dataset)
            loss_all = 0
            loss_nll_x_all = 0
            loss_nll_y_all = 0
            loss_velocity_all = 0
            loss_velocity_consistency_all = 0
            loss_kl_z0_all = 0
            loss_kl_t_all = 0

            for idx_batch, (x, y, vx, v_mask, w) in enumerate(dataloader):

                x, y, vx, v_mask, w = x.to(self.device), y.to(self.device), vx.to(self.device), v_mask.to(self.device), w.to(self.device)
                x_pred, y_pred, vx_pred, vy_pred, z0_mean, z0_std, alpha_t, beta_t = self(x, y)

                #cell weights
                w = w/torch.sum(w)
                w = w.reshape([w.shape[0], 1])

                qz0 = distributions.Normal(loc=z0_mean, scale=z0_std)
                qt = distributions.Beta(concentration1=alpha_t, concentration0=beta_t)
                likelihood_x = distributions.Normal(loc=x_pred, scale=self.sigma_x)
                likelihood_y = distributions.Normal(loc=y_pred, scale=self.sigma_y)
                nll_x = -(w*likelihood_x.log_prob(x)).sum(dim=1).sum(dim=0)
                nll_y = -(w*likelihood_y.log_prob(y)).sum(dim=1).sum(dim=0)
                kl_z0 = distributions.kl_divergence(qz0, self.pz0).sum(dim=1).mean(dim=0)
                kl_t = distributions.kl_divergence(qt, self.pt).sum(dim=1).mean(dim=0)

                vx_pred_masked = vx_pred * v_mask
                vel_loss = -torch.mean(COSLoss(vx_pred_masked, vx))            
                con_loss = torch.mean((pairwise_cosine_similarity(vx_pred) - pairwise_cosine_similarity(vy_pred))**2)

                loss_nll = nll_x + nll_y
                loss_velocity = self.k_velocity * vel_loss 
                loss_velocity_consistency = self.k_consistency * con_loss
                loss_kl = self.k_z0 * kl_z0 + self.k_t * kl_t
                loss = loss_nll + loss_velocity + loss_velocity_consistency + loss_kl

                # for logging
                loss_all += loss.detach().cpu().numpy() * x.shape[0]
                loss_nll_x_all += nll_x.detach().cpu().numpy() * x.shape[0]
                loss_nll_y_all += nll_y.detach().cpu().numpy() * x.shape[0]
                loss_velocity_all += vel_loss.detach().cpu().numpy() * x.shape[0]
                loss_velocity_consistency_all += con_loss.detach().cpu().numpy() * x.shape[0]
                loss_kl_z0_all += kl_z0.detach().cpu().numpy() * x.shape[0]
                loss_kl_t_all += kl_t.detach().cpu().numpy() * x.shape[0]

        return loss_all/N, loss_nll_x_all/N, loss_nll_y_all/N, loss_velocity_all/N, loss_velocity_consistency_all/N, loss_kl_z0_all/N, loss_kl_t_all/N


    def fit(self, dataloader_train, dataloader_test, optimizer, max_epoch):
        """
        Training the model for the full epochs.

        Parameters
        -------
        dataloader_train: Pytorch dataloader for train data
        dataloader_test: Pytorch dataloader for test data
        optimizer: model optimizer
        max_epoch: maximum number of epochs to train the model
        """

        lr_scheduler = LRScheduler(optimizer, patience=5, factor=0.5)
        early_stopping = EarlyStopping(patience=10)
        self.logger.start()

        for epoch in range(1, 1+max_epoch):

            self.mode = 'train'

            loss_train = self.train_step(dataloader_train, optimizer)
            loss_test, loss_nll_x_test, loss_nll_y_test, loss_velocity_test, loss_velocity_consistency_test, loss_kl_z0_test, loss_kl_t_test = self.test_step(dataloader_test)

            self.logger.add([epoch, loss_train, loss_test, loss_nll_x_test, loss_nll_y_test, loss_velocity_test, loss_velocity_consistency_test, loss_kl_z0_test, loss_kl_t_test])
            self.logger.save(f"../log/{self.model_suffix}.log")

            # Early stopping
            lr_scheduler(loss_test)
            early_stopping(loss_test)
            if loss_test <= early_stopping.best_loss:
                torch.save({
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, self.ckpt_path)
            if early_stopping.early_stop:
                break


    def load(self, optimizer):
        if os.path.exists(self.ckpt_path):
            checkpoint = torch.load(self.ckpt_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(self.ckpt_path))
        else:
            print('Model does not exist.')


    def evaluate(self, adata_rna, adata_atac, dataloader, n_samples=100):
        """
        Evaluates the model and adds the predicted quantities to the adata.

        Parameters
        ----------
        adata_rna: RNA adata
        adata_atac: TF motif adata
        dataloader: Pytorch dataloader for all data
        n_samples: number of samples per cell to generate from the posterior
        distributions of z0 and t.

        Returns
        -------
        adata_rna_pred: updated adata_rna containing the predicted quantities
        adata_atac_pred: updated adata_atac containing the predicted quantities
        """

        batch_size = dataloader.batch_size
        zx_dim = self.zx_dim
        zy_dim = self.zy_dim
        z_dim = zx_dim + zy_dim
        device = self.device
        n_cells = adata_rna.shape[0]
        n_genes = adata_rna.shape[1]
        n_tfs = adata_atac.shape[1]

        z0_mean = np.zeros([n_cells, z_dim])
        z = np.zeros([n_samples, n_cells, z_dim])
        vz = np.zeros([n_samples, n_cells, z_dim])

        x_obs = np.zeros([n_cells, n_genes])
        vx_obs = np.zeros([n_cells, n_genes])
        x_pred = np.zeros([n_samples, n_cells, n_genes])
        vx_pred = np.zeros([n_samples, n_cells, n_genes])

        y_obs = np.zeros([n_cells, n_tfs])
        y_pred = np.zeros([n_samples, n_cells, n_tfs])
        vy_pred = np.zeros([n_samples, n_cells, n_tfs])

        latent_time_mean = np.zeros(n_cells)
        latent_time_std = np.zeros(n_cells)

        with torch.no_grad():
            self.eval()
            for n in range(n_samples):
                print('n: ', n)
                for idx_batch, (x, y, vx, v_mask, w) in enumerate(dataloader):
                    x, y, vx = x.to(device), y.to(device), vx.to(device)
                    #z0_mean_, z_, vz_, x_pred_, y_pred_, vx_pred_, vy_pred_, latent_time_, latent_time_std_ = self(x, y)
                    x_pred_, y_pred_, vx_pred_, vy_pred_, z0_mean_, z_, vz_, latent_time_, latent_time_std_ = self(x, y)

                    if n == 0:
                        x_obs[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = x.detach().cpu().numpy()
                        vx_obs[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vx.detach().cpu().numpy()
                        y_obs[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = y.detach().cpu().numpy()
                        z0_mean[idx_batch*batch_size:idx_batch*batch_size+batch_size] = z0_mean_.detach().cpu().numpy()
                        latent_time_mean[idx_batch*batch_size:idx_batch*batch_size+batch_size] = latent_time_.detach().cpu().numpy()
                        latent_time_std[idx_batch*batch_size:idx_batch*batch_size+batch_size] = latent_time_std_.detach().cpu().numpy()

                    z[n, idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = z_.detach().cpu().numpy()
                    vz[n, idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vz_.detach().cpu().numpy()
                    vx_pred[n, idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vx_pred_.detach().cpu().numpy()
                    vy_pred[n, idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vy_pred_.detach().cpu().numpy()
                    x_pred[n, idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = x_pred_.detach().cpu().numpy()
                    y_pred[n, idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = y_pred_.detach().cpu().numpy()

        # std
        z_std = np.std(z, axis=0)
        vz_std = np.std(vz, axis=0)
        vx_pred_std = np.std(vx_pred, axis=0)
        vy_pred_std = np.std(vy_pred, axis=0)
        x_pred_std = np.std(x_pred, axis=0)
        y_pred_std = np.std(y_pred, axis=0)

        # mean
        z_mean = np.mean(z, axis=0)
        vz_mean = np.mean(vz, axis=0)
        vx_pred_mean = np.mean(vx_pred, axis=0)
        vy_pred_mean = np.mean(vy_pred, axis=0)
        x_pred_mean = np.mean(x_pred, axis=0)
        y_pred_mean = np.mean(y_pred, axis=0)

        # zx/zy mean
        zx_mean = z_mean[:,:zx_dim]
        vzx_mean = vz_mean[:,:zx_dim]
        zy_mean = z_mean[:,zx_dim:(zx_dim+zy_dim)]
        vzy_mean = vz_mean[:,zx_dim:(zx_dim+zy_dim)]
        zx0_mean = z0_mean[:,:zx_dim]
        zy0_mean = z0_mean[:,zx_dim:]

        # zx/zy std
        zx_std = z_std[:,:zx_dim]
        vzx_std = vz_std[:,:zx_dim]
        zy_std = z_std[:,zx_dim:(zx_dim+zy_dim)]
        vzy_std = vz_std[:,zx_dim:(zx_dim+zy_dim)]

        # sorted latent time between [0,1]
        latent_time_sorted = latent_time_mean.copy()
        idx_sorted = np.argsort(latent_time_sorted)
        latent_time_sorted[idx_sorted] = np.arange(len(latent_time_mean))/(len(latent_time_mean)-1)

        # add to adata_rna
        adata_rna_pred = adata_rna.copy()
        adata_atac_pred = adata_atac.copy()

        adata_rna_pred.obs['latent_time_scvelo'] = adata_rna_pred.obs['latent_time'].copy()
        adata_atac_pred.obs['latent_time_scvelo'] = adata_rna_pred.obs['latent_time'].copy()

        adata_rna_pred.obs['latent_time_mean'] = latent_time_mean
        adata_rna_pred.obs['latent_time_std'] = latent_time_std
        adata_rna_pred.obs['latent_time'] = latent_time_sorted
        
        adata_atac_pred.obs['latent_time_mean'] = latent_time_mean
        adata_atac_pred.obs['latent_time_std'] = latent_time_std
        adata_atac_pred.obs['latent_time'] = latent_time_sorted

        adata_rna_pred.obsm['z0_mean'] = z0_mean
        adata_rna_pred.obsm['z_mean'] = z_mean
        adata_rna_pred.obsm['vz_mean'] = vz_mean
        adata_rna_pred.layers['x_pred_mean'] = x_pred_mean
        adata_rna_pred.layers['x_pred_std'] = x_pred_std
        adata_rna_pred.layers['vx_pred_mean'] = vx_pred_mean
        adata_rna_pred.layers['vx_pred_std'] = vx_pred_std

        adata_atac_pred.obsm['z0_mean'] = z0_mean
        adata_atac_pred.obsm['z_mean'] = z_mean
        adata_atac_pred.obsm['vz_mean'] = vz_mean
        adata_atac_pred.layers['y_pred_mean'] = y_pred_mean
        adata_atac_pred.layers['y_pred_std'] = y_pred_std
        adata_atac_pred.layers['vy_pred_mean'] = vy_pred_mean
        adata_atac_pred.layers['vy_pred_std'] = vy_pred_std

        adata_rna_pred.obsm['zx0_mean'] = zx0_mean
        adata_rna_pred.obsm['zx_mean'] = zx_mean
        adata_rna_pred.obsm['zx_std'] = zx_std
        adata_rna_pred.obsm['vzx_mean'] = vzx_mean
        adata_rna_pred.obsm['vzx_std'] = vzx_std

        adata_atac_pred.obsm['zy0_mean'] = zy0_mean
        adata_atac_pred.obsm['zy_mean'] = zy_mean
        adata_atac_pred.obsm['zy_std'] = zy_std
        adata_atac_pred.obsm['vzy_mean'] = vzy_mean
        adata_atac_pred.obsm['vzy_std'] = vzy_std

        return adata_rna_pred, adata_atac_pred


    def calculate_jacobians(self, adata_rna_pred, adata_atac_pred, dataloader, genes_of_interest, epsilon=1e-4):
        """
        Calculates the Jacobians of the model and adds them to the adata.

        Parameters
        ----------
        adata_rna_pred: RNA adata
        adata_atac_pred: TF motif adata
        dataloader: Pytorch dataloader for all data
        genes_of_interest: genes for which to calculate the Jacobians
        epsilon: a small number used to calculate the Jacobians

        Returns
        -------
        adata_rna_pred: updated adata_rna containing the calculated Jacobians
        """

        batch_size = dataloader.batch_size
        device = self.device
        n_cells = adata_rna_pred.shape[0]
        n_genes = adata_rna_pred.shape[1]
        n_tfs = adata_atac_pred.shape[1]
        tfs = list(adata_atac_pred.var['TF'].values)

        genes = adata_rna_pred.var_names.values
        n_gi = len(genes_of_interest)
        genes_of_interest_idx = np.zeros(n_gi).astype(np.int32)
        for idx, g in enumerate(genes_of_interest):
            genes_of_interest_idx[idx] = np.where(genes==g)[0].item()

        J_dvx_dx = np.zeros([n_cells, n_gi, n_gi])
        J_dvy_dx = np.zeros([n_cells, n_tfs, n_gi])
        J_dvx_dy = np.zeros([n_cells, n_gi, n_tfs])
        J_dvy_dy = np.zeros([n_cells, n_tfs, n_tfs])

        with torch.no_grad():
            self.eval()
            for idx_batch, (x, y, vx, v_mask, w) in enumerate(dataloader):
                print(f'Batch {idx_batch+1}/{len(dataloader)}')
                x, y = x.to(device), y.to(device)
                x_pred, y_pred, vx_pred, vy_pred, z0_mean, z, vz, latent_time, latent_time_std = self(x, y)

                for i, g in enumerate(genes_of_interest_idx):
                    e = np.zeros(n_genes)
                    e[g] = 1.
                    x_epsilon = x + epsilon * torch.tensor(e).type(torch.float32).to(device)
                    x_pred, y_pred, vx_pred_epsilon, vy_pred_epsilon, z0_mean, z, vz, latent_time, latent_time_std = self(x_epsilon, y)

                    J_dvx_dx[idx_batch*batch_size:idx_batch*batch_size+batch_size, :, i] = (1/epsilon) * (vx_pred_epsilon.detach().cpu().numpy()[:, genes_of_interest_idx] - vx_pred.detach().cpu().numpy()[:, genes_of_interest_idx])
                    J_dvy_dx[idx_batch*batch_size:idx_batch*batch_size+batch_size, :, i] = (1/epsilon) * (vy_pred_epsilon.detach().cpu().numpy() - vy_pred.detach().cpu().numpy())

                for t in range(n_tfs):
                    e = np.zeros(n_tfs)
                    e[t] = 1.
                    y_epsilon = y + epsilon * torch.tensor(e).type(torch.float32).to(device)
                    x_pred, y_pred, vx_pred_epsilon, vy_pred_epsilon, z0_mean, z, vz, latent_time, latent_time_std = self(x, y_epsilon)

                    J_dvx_dy[idx_batch*batch_size:idx_batch*batch_size+batch_size, :, t] = (1/epsilon) * (vx_pred_epsilon.detach().cpu().numpy()[:, genes_of_interest_idx] - vx_pred.detach().cpu().numpy()[:, genes_of_interest_idx])
                    J_dvy_dy[idx_batch*batch_size:idx_batch*batch_size+batch_size, :, t] = (1/epsilon) * (vy_pred_epsilon.detach().cpu().numpy() - vy_pred.detach().cpu().numpy())

        # save jacobians to adata
        adata_rna_pred.obsm['J_dvx_dx'] = J_dvx_dx
        adata_rna_pred.obsm['J_dvy_dx'] = J_dvy_dx
        adata_rna_pred.obsm['J_dvx_dy'] = J_dvx_dy
        adata_rna_pred.obsm['J_dvy_dy'] = J_dvy_dy
        adata_rna_pred.uns['Jacobians:genes_of_interest'] = genes_of_interest
        adata_rna_pred.uns['Jacobians:tfs'] = tfs

        return adata_rna_pred


    def predict_perturbation(self, adata_rna_pred, adata_atac_pred, dataloader, perturbed_genes):
        """
        Performs in-silico perturbations and adds them to the adata

        Parameters
        ----------
        adata_rna_pred: RNA adata
        adata_atac_pred: TF motif adata
        dataloader: Pytorch dataloader for all data
        perturbed_genes: genes to perturb

        Returns
        -------
        adata_rna_pred: updated adata_rna containing the post-perturbation delta velocities 
        """

        batch_size = dataloader.batch_size
        zx_dim = self.zx_dim
        zy_dim = self.zy_dim
        z_dim = zx_dim + zy_dim
        device = self.device
        n_cells = adata_rna_pred.shape[0]
        n_genes = adata_rna_pred.shape[1]
        n_tfs = adata_atac_pred.shape[1]
        n_pt_genes = len(perturbed_genes)

        vz = np.zeros([n_cells, z_dim])
        vx_pred = np.zeros([n_cells, n_genes])
        vy_pred = np.zeros([n_cells, n_tfs])
        latent_time_mean = np.zeros(n_cells)

        vz_perturbed = np.zeros([n_cells, z_dim])
        vx_pred_perturbed = np.zeros([n_cells, n_genes])
        vy_pred_perturbed = np.zeros([n_cells, n_tfs])
        latent_time_mean_perturbed = np.zeros(n_cells)

        delta_vx = np.zeros([n_cells, n_genes, n_pt_genes])
        delta_vy = np.zeros([n_cells, n_tfs, n_pt_genes])
        delta_vz = np.zeros([n_cells, z_dim, n_pt_genes])
        delta_latent_time = np.zeros([n_cells, n_pt_genes])

        TFs = adata_atac_pred.var['TF'].values
        adata_atac_pred.var_names = TFs

        with torch.no_grad():
            self.eval()
            for idx_batch, (x, y, vx, v_mask, w) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                x_pred_, y_pred_, vx_pred_, vy_pred_, z0_mean_, z_, vz_, latent_time_, latent_time_std_ = self(x, y)

                latent_time_mean[idx_batch*batch_size:idx_batch*batch_size+batch_size] = latent_time_.detach().cpu().numpy()
                vz[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vz_.detach().cpu().numpy()
                vx_pred[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vx_pred_.detach().cpu().numpy()
                vy_pred[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vy_pred_.detach().cpu().numpy()

            for idx, g in enumerate(perturbed_genes):
                print(f'idx: {idx} / perturbed gene: {g}')
                gene_idx = np.where(adata_rna_pred.var_names==g)[0].item()
                x_gene_min = adata_rna_pred.X[:,gene_idx].min()
                if g in TFs:
                    tf_idx = np.where(adata_atac_pred.var_names==g)[0].item()
                    y_tf_min = adata_atac_pred.X[:,tf_idx].min()

                for idx_batch, (x, y, vx, v_mask, w) in enumerate(dataloader):
                    x[:,gene_idx] = torch.tensor(x_gene_min, dtype=torch.float32)
                    if g in TFs:
                        y[:,tf_idx] = torch.tensor(y_tf_min, dtype=torch.float32)

                    x, y = x.to(device), y.to(device)
                    x_pred_, y_pred_, vx_pred_, vy_pred_, z0_mean_, z_, vz_, latent_time_, latent_time_std_ = self(x, y)

                    latent_time_mean_perturbed[idx_batch*batch_size:idx_batch*batch_size+batch_size] = latent_time_.detach().cpu().numpy()
                    vz_perturbed[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vz_.detach().cpu().numpy()
                    vx_pred_perturbed[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vx_pred_.detach().cpu().numpy()
                    vy_pred_perturbed[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vy_pred_.detach().cpu().numpy()

                # delta
                delta_vx[:,:,idx] = vx_pred_perturbed - vx_pred
                delta_vy[:,:,idx] = vy_pred_perturbed - vy_pred
                delta_vz[:,:,idx] = vz_perturbed - vz
                delta_latent_time[:,idx] = latent_time_mean_perturbed - latent_time_mean

        # delta_vzx/delta_vzy
        delta_vzx = delta_vz[:,:zx_dim,:]
        delta_vzy = delta_vz[:,zx_dim:,:]

        # save deltas to adata
        adata_rna_pred.obsm['delta_vx'] = delta_vx
        adata_rna_pred.obsm['delta_vy'] = delta_vy
        adata_rna_pred.obsm['delta_vzx'] = delta_vzx
        adata_rna_pred.obsm['delta_vzy'] = delta_vzy
        adata_rna_pred.obsm['delta_latent_time'] = delta_latent_time
        adata_rna_pred.uns['Perturbed_genes'] = perturbed_genes

        return adata_rna_pred
