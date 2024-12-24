# Author: Alireza Karbalayghareh

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torch import distributions
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


class RNADataset(Dataset):
    """
    A custom dataset class for scRNA data.

    Parameters
    ----------
    adata_rna: an anndata object containing RNA expression.
    """
    def __init__(self, adata_rna):

        # Process RNA data
        self.x_rna = self._to_dense(adata_rna.X)
        self.vx = np.nan_to_num(self._to_dense(adata_rna.layers['velocity']))

        # Store observations and variables for RNA and ATAC
        self.rna_obs = adata_rna.obs
        self.rna_var = adata_rna.var

        # Create a mask for velocity genes
        self.velocity_genes_mask = (~np.isnan(adata_rna.layers['velocity'])).astype(np.float32)

        # Map cell types to indices
        self.celltype_indices = {celltype: np.where(self.rna_obs['final.celltype'] == celltype)[0]
                                 for celltype in self.rna_obs['final.celltype'].unique()}

        # Calculate weights for cell types based on their frequencies
        self.weights_dict = self._calculate_weights(adata_rna.obs['final.celltype'])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.X_rna.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset.
        """
        x = torch.tensor(self.x_rna[idx, :], dtype=torch.float32)
        vx = torch.tensor(self.vx[idx, :] * self.velocity_genes_mask[idx, :], dtype=torch.float32)
        v_mask = torch.tensor(self.velocity_genes_mask[idx, :], dtype=torch.float32)
        celltype = self.rna_obs['final.celltype'][idx]
        weight = torch.tensor(self.weights_dict[celltype], dtype=torch.float32)

        return x, vx, v_mask, weight

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


class node_func_rna(nn.Module):
    """
    Neural ODE function (deep) in the latent space.
    It returns the latent velocities.

    Parameters
    ----------
    zx_dim: dimension of RNA latent space
    num_layer: number of layers
    """

    def __init__(self, zx_dim, num_layer):
        super().__init__()

        self.zx_dim = zx_dim
        self.num_layer = num_layer
        layers = []
        for l in range(num_layer):
            if l < self.num_layer - 1:
                layers.append(nn.Linear(zx_dim, zx_dim, bias=True))
                layers.append(nn.GELU())
            else:
                layers.append(nn.Linear(zx_dim, zx_dim, bias=True))
        
        self.node_net = nn.Sequential(*layers)

    def forward(self, t, z):
        return self.node_net(z)


class node_func_wide_rna(nn.Module):
    """
    Neural ODE function (wide) in the latent space.
    It returns the latent velocities.

    Parameters
    ----------
    zx_dim: dimension of RNA latent space
    num_hidden: number of hidden neurons in the middle layer
    """

    def __init__(self, zx_dim, num_hidden):
        super().__init__()

        self.fzx = nn.Sequential(
                                nn.Linear(zx_dim, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, zx_dim),
                                )

        for m in self.fzx.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)

    def forward(self, t, z):
        return self.fzx(z)


class node_func_wide_multi_lineage_rna(nn.Module):
    """
    Neural ODE function (wide, multi lineage) in the latent space.
    It returns the latent velocities.

    Parameters
    ----------
    zx_dim: dimension of RNA latent space
    h_dim: dimension of augmented space
    num_hidden: number of hidden neurons in the middle layer
    device: GPU device
    """

    def __init__(self, zx_dim, h_dim, num_hidden, device):
        super().__init__()

        self.z_dim = zx_dim
        self.h_dim = h_dim
        self.device = device
        self.node_net = nn.Sequential(
                                nn.Linear(self.z_dim+self.h_dim, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, self.z_dim),
                                )
        
        for m in self.node_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0.0)

    def forward(self, t, zh):

        z = zh[..., :self.z_dim]
        h = zh[..., self.z_dim:]

        fz = self.node_net(torch.cat((z, h), dim=-1))

        return torch.cat((fz, torch.zeros(h.shape).to(self.device)), dim=-1)


class node_func_wide_time_variant_rna(nn.Module):
    """
    Neural ODE function (wide, time-variant) in the latent space.
    It returns the latent velocities.

    Parameters
    ----------
    zx_dim: dimension of RNA latent space
    num_hidden: number of hidden neurons in the middle layer
    """

    def __init__(self, zx_dim, num_hidden):
        super().__init__()

        self.node_net = nn.Sequential(
                                nn.Linear(zx_dim+1, num_hidden),
                                nn.GELU(),
                                nn.Linear(num_hidden, zx_dim)
                                )
    def forward(self, t, z):
        if t.dim() == 0:
            t = torch.full_like(z, fill_value=t.item())[:,0:1]
        return self.node_net(torch.cat((t, z), dim=-1))


class DynaVelo_RNA(nn.Module):
    """
    DynaVelo model with RNA input.

    Parameters
    ----------
    zx_dim: dimension of RNA latent space
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
    seed: random seed
    time_variant: if the neural ODE should be time variant
    mode: select the mode (train, evaluation-sample, evaluation-fixed)
    device: select the GPU
    dataset_name: enter name of the dataset
    sample_name: enter name of the sample
    """

    def __init__(self, x_dim,
                 zx_dim=50, num_hidden=200, k_t=1000, k_z0=1000, 
                 k_velocity=10000, mu_pz0=0, sigma_pz0=1,
                 alpha_pt=2, beta_pt=2, sigma_x=0.1, seed=0,
                 time_variant=False, mode='train', device='cuda:0',
                 dataset_name='GCB', sample_name='CtcfWT29'):
        super().__init__()

        self.x_dim = x_dim
        self.zx_dim = zx_dim
        self.num_hidden = num_hidden
        self.k_t = k_t
        self.k_z0 = k_z0
        self.k_velocity = k_velocity
        self.seed = seed
        self.time_variant = time_variant
        self.mode = mode
        self.device = device
        self.dataset_name = dataset_name
        self.sample_name = sample_name
        self.logger = Logger(["Epoch", "loss_train", "loss_test", "nll_x",  
                              "loss_vel", "kl_z0", "kl_t"], verbose=True)

        mu_pz0 = torch.tensor(mu_pz0).to(device)
        sigma_pz0 = torch.tensor(sigma_pz0).to(device)
        self.pz0 = distributions.Normal(loc=mu_pz0, scale=sigma_pz0)

        alpha_pt = torch.tensor(alpha_pt).to(device)
        beta_pt = torch.tensor(beta_pt).to(device)
        self.pt = distributions.Beta(concentration1=alpha_pt, concentration0=beta_pt)

        self.sigma_x = torch.tensor(sigma_x).to(device)

        self.encoder_net_x = nn.Sequential(
                                        nn.Linear(x_dim, zx_dim),
                                        nn.GELU(),
                                        )
        
        self.encoder_net_tx = nn.Sequential(
                                        nn.Linear(x_dim, zx_dim),
                                        nn.GELU(),
                                        )
        
        self.decoder_net_x = nn.Sequential(
                                        nn.Linear(zx_dim, x_dim),
                                        nn.ReLU()
                                        )

        self.mu_zx0_net = nn.Sequential(
                                        nn.Linear(zx_dim, zx_dim),
                                        )
        
        self.sigma_zx0_net = nn.Sequential(
                                        nn.Linear(zx_dim, zx_dim),
                                        nn.Softplus()
                                        )

        self.alpha_t_net = nn.Sequential(
                                        nn.Linear(zx_dim, 1),
                                        nn.Softplus()
                                        )

        self.beta_t_net = nn.Sequential(
                                        nn.Linear(zx_dim, 1),
                                        nn.Softplus()
                                        )
        
        if self.time_variant:
            self.node_func = node_func_wide_time_variant_rna(zx_dim, num_hidden)
        else:
            self.node_func = node_func_wide_rna(zx_dim, num_hidden)


    def forward(self, x):
        """
        A forward pass through the DynaVelo model.

        Parameters
        ----------
        x: RNA expression matrix with the shape (n_cells_in_batch, n_genes)

        Returns
        -------
        x_pred: predicted RNA expression values
        vx_pred: predicted RNA velocities for all input genes
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
            z0_mean = self.mu_zx0_net(hx)
            z0_std = self.sigma_zx0_net(hx)
            z0 = z0_mean + z0_std * torch.randn(z0_std.shape).to(self.device)

            ht = self.encoder_net_tx(x)
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

            x_pred = self.decoder_net_x(z)

            # Jacobians:
            Jx = vmap(jacrev(self.decoder_net_x))(z)

            vx_pred = torch.bmm(Jx, vz.unsqueeze(2)).squeeze()

            return x_pred, vx_pred, z0_mean, z0_std, alpha_t, beta_t

        elif self.mode == 'evaluation-fixed':

            hx = self.encoder_net_x(x)
            z0_mean = self.mu_zx0_net(hx)
            z0 = z0_mean

            ht = self.encoder_net_tx(x)
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

            x_pred = self.decoder_net_x(z)

            # Jacobians:
            Jx = vmap(jacrev(self.decoder_net_x))(z)

            vx_pred = torch.bmm(Jx, vz.unsqueeze(2)).squeeze()

            latent_time = t_mean
            latent_time_std = torch.sqrt((alpha_t*beta_t)/(((alpha_t+beta_t)**2)*(alpha_t+beta_t+1))).squeeze()
            
            return x_pred, vx_pred, z0_mean, z, vz, latent_time, latent_time_std
        
        elif self.mode == 'evaluation-sample':

            hx = self.encoder_net_x(x)
            z0_mean = self.mu_zx0_net(hx)
            z0_std = self.sigma_zx0_net(hx)
            z0 = z0_mean + z0_std * torch.randn(z0_std.shape).to(self.device)

            ht = self.encoder_net_tx(x)
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

            x_pred = self.decoder_net_x(z)

            # Jacobians:
            Jx = vmap(jacrev(self.decoder_net_x))(z)

            vx_pred = torch.bmm(Jx, vz.unsqueeze(2)).squeeze()

            latent_time = t_mean
            latent_time_std = torch.sqrt((alpha_t*beta_t)/(((alpha_t+beta_t)**2)*(alpha_t+beta_t+1))).squeeze()

            return x_pred, vx_pred, z0_mean, z, vz, latent_time, latent_time_std


    def generate_trajectory(self, x, t_max, n_points):
        """
        Generates synthetic trajectories using the trained DynaVelo models.

        Parameters
        ----------
        x: a terminal cell's RNA expression with the shape (1, n_genes)
        t_max: generate trajectory from time 0 to t_max
        n_points: number of synthetic cells to generate

        Returns
        -------
        x_pred: generated RNA expression with the shape (n_points, n_genes)
        vx_pred: generated RNA velocity with the shape (n_points, n_genes)
        t_interval: time interval of the generated trajectory (0, t_max)
        """

        hx = self.encoder_net_x(x)
        mu_zx0 = self.mu_zx0_net(hx)
        sigma_zx0 = self.sigma_zx0_net(hx)
        z0 = mu_zx0 + sigma_zx0 * torch.randn(sigma_zx0.shape).to(self.device)

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

        x_pred = self.decoder_net_x(z)

        # Jacobians:
        Jx = vmap(jacrev(self.decoder_net_x))(z)

        vx_pred = torch.bmm(Jx, vz.unsqueeze(2)).squeeze()

        return x_pred, vx_pred, t_interval


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

        for idx_batch, (x, vx, v_mask, w) in enumerate(dataloader):

            x, vx, v_mask, w = x.to(self.device), vx.to(self.device), v_mask.to(self.device), w.to(self.device)
            x_pred, vx_pred, z0_mean, z0_std, alpha_t, beta_t = self(x)

            optimizer.zero_grad()

            #cell weights
            w = w/torch.sum(w)
            w = w.reshape([w.shape[0], 1])

            qz0 = distributions.Normal(loc=z0_mean, scale=z0_std)
            qt = distributions.Beta(concentration1=alpha_t, concentration0=beta_t)
            likelihood_x = distributions.Normal(loc=x_pred, scale=self.sigma_x)
            nll_x = -(w*likelihood_x.log_prob(x)).sum(dim=1).sum(dim=0)
            kl_z0 = distributions.kl_divergence(qz0, self.pz0).sum(dim=1).mean(dim=0)
            kl_t = distributions.kl_divergence(qt, self.pt).sum(dim=1).mean(dim=0)

            vx_pred_masked = vx_pred * v_mask
            vel_loss = -torch.mean(COSLoss(vx_pred_masked, vx))            

            loss_nll = nll_x
            loss_velocity = self.k_velocity * vel_loss 
            loss_kl = self.k_z0 * kl_z0 + self.k_t * kl_t
            loss = loss_nll + loss_velocity + loss_kl

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
        loss_velocity_all: RNA velocity loss
        loss_kl_z0_all: KL divergence of z0
        loss_kl_t_all: KL divergence of t
        """

        with torch.no_grad():
            self.eval()
            COSLoss = nn.CosineSimilarity(dim=-1, eps=1e-08)
            N = len(dataloader.dataset)
            loss_all = 0
            loss_nll_x_all = 0
            loss_velocity_all = 0
            loss_kl_z0_all = 0
            loss_kl_t_all = 0

            for idx_batch, (x, vx, v_mask, w) in enumerate(dataloader):

                x, vx, v_mask, w = x.to(self.device), vx.to(self.device), v_mask.to(self.device), w.to(self.device)
                x_pred, vx_pred, z0_mean, z0_std, alpha_t, beta_t = self(x)

                #cell weights
                w = w/torch.sum(w)
                w = w.reshape([w.shape[0], 1])

                qz0 = distributions.Normal(loc=z0_mean, scale=z0_std)
                qt = distributions.Beta(concentration1=alpha_t, concentration0=beta_t)
                likelihood_x = distributions.Normal(loc=x_pred, scale=self.sigma_x)
                nll_x = -(w*likelihood_x.log_prob(x)).sum(dim=1).sum(dim=0)
                kl_z0 = distributions.kl_divergence(qz0, self.pz0).sum(dim=1).mean(dim=0)
                kl_t = distributions.kl_divergence(qt, self.pt).sum(dim=1).mean(dim=0)

                vx_pred_masked = vx_pred * v_mask
                vel_loss = -torch.mean(COSLoss(vx_pred_masked, vx))            

                loss_nll = nll_x
                loss_velocity = self.k_velocity * vel_loss 
                loss_kl = self.k_z0 * kl_z0 + self.k_t * kl_t
                loss = loss_nll + loss_velocity

                # for logging
                loss_all += loss.detach().cpu().numpy() * x.shape[0]
                loss_nll_x_all += nll_x.detach().cpu().numpy() * x.shape[0]
                loss_velocity_all += vel_loss.detach().cpu().numpy() * x.shape[0]
                loss_kl_z0_all += kl_z0.detach().cpu().numpy() * x.shape[0]
                loss_kl_t_all += kl_t.detach().cpu().numpy() * x.shape[0]

        return loss_all/N, loss_nll_x_all/N, loss_velocity_all/N, loss_kl_z0_all/N, loss_kl_t_all/N


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

        train_dir = f'../checkpoints/{self.dataset_name}/{self.sample_name}/'
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        model_suffix = f'{self.dataset_name}_{self.sample_name}_DynaVelo_RNA_num_hidden_{self.num_hidden}_zxdim_{self.zx_dim}_k_z0_{str(self.k_z0)}_k_t_{str(self.k_t)}_k_velocity_{str(self.k_velocity)}_seed_{self.seed}'
        ckpt_path = train_dir+model_suffix+'.pth'

        lr_scheduler = LRScheduler(optimizer, patience=5, factor=0.5)
        early_stopping = EarlyStopping(patience=10)
        self.logger.start()

        for epoch in range(1, 1+max_epoch):

            self.mode = 'train'

            loss_train = self.train_step(dataloader_train, optimizer)
            loss_test, loss_nll_x_test, loss_velocity_test, loss_kl_z0_test, loss_kl_t_test = self.test_step(dataloader_test)

            self.logger.add([epoch, loss_train, loss_test, loss_nll_x_test, loss_velocity_test, loss_kl_z0_test, loss_kl_t_test])
            self.logger.save(f"../log/{model_suffix}.log")

            # Early stopping
            lr_scheduler(loss_test)
            early_stopping(loss_test)
            if loss_test <= early_stopping.best_loss:
                torch.save({
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, ckpt_path)
            if early_stopping.early_stop:
                break


    def evaluate(self, adata_rna, dataloader, n_samples=100):
        """        adata_atac_pred: updated adata_atac containing the predicted quantities

        Evaluates the model and adds the predicted quantities to the adata

        Parameters
        ----------
        adata_rna: RNA adata
        dataloader: Pytorch dataloader for all data
        n_samples: number of samples per cell to generate from the posterior
        distributions of z0 and t.

        Returns
        -------
        adata_rna_pred: updated adata_rna containing the predicted quantities
        """

        batch_size = dataloader.batch_size
        z_dim = self.zx_dim
        device = self.device
        n_cells = adata_rna.shape[0]
        n_genes = adata_rna.shape[1]

        z0_mean = np.zeros([n_cells, z_dim])
        z = np.zeros([n_samples, n_cells, z_dim])
        vz = np.zeros([n_samples, n_cells, z_dim])

        x_obs = np.zeros([n_cells, n_genes])
        vx_obs = np.zeros([n_cells, n_genes])
        x_pred = np.zeros([n_samples, n_cells, n_genes])
        vx_pred = np.zeros([n_samples, n_cells, n_genes])

        latent_time_mean = np.zeros(n_cells)
        latent_time_std = np.zeros(n_cells)

        with torch.no_grad():
            self.eval()
            for n in range(n_samples):
                print('n: ', n)
                for idx_batch, (x, vx, v_mask, w) in enumerate(dataloader):
                    x, vx = x.to(device), vx.to(device)
                    z0_mean_, z_, vz_, x_pred_, vx_pred_, latent_time_, latent_time_std_ = self(x)

                    if n == 0:
                        x_obs[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = x.detach().cpu().numpy()
                        vx_obs[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vx.detach().cpu().numpy()
                        z0_mean[idx_batch*batch_size:idx_batch*batch_size+batch_size] = z0_mean_.detach().cpu().numpy()
                        latent_time_mean[idx_batch*batch_size:idx_batch*batch_size+batch_size] = latent_time_.detach().cpu().numpy()
                        latent_time_std[idx_batch*batch_size:idx_batch*batch_size+batch_size] = latent_time_std_.detach().cpu().numpy()

                    z[n, idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = z_.detach().cpu().numpy()
                    vz[n, idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vz_.detach().cpu().numpy()
                    vx_pred[n, idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vx_pred_.detach().cpu().numpy()
                    x_pred[n, idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = x_pred_.detach().cpu().numpy()

        # std
        z_std = np.std(z, axis=0)
        vz_std = np.std(vz, axis=0)
        vx_pred_std = np.std(vx_pred, axis=0)
        x_pred_std = np.std(x_pred, axis=0)

        # mean
        z_mean = np.mean(z, axis=0)
        vz_mean = np.mean(vz, axis=0)
        vx_pred_mean = np.mean(vx_pred, axis=0)
        x_pred_mean = np.mean(x_pred, axis=0)

        # sorted latent time between [0,1]
        latent_time_sorted = latent_time_mean.copy()
        idx_sorted = np.argsort(latent_time_sorted)
        latent_time_sorted[idx_sorted] = np.arange(len(latent_time_mean))/(len(latent_time_mean)-1)

        # add to adata_rna
        adata_rna_pred = adata_rna.copy()

        adata_rna_pred.obs['latent_time_scvelo'] = adata_rna_pred.obs['latent_time'].copy()

        adata_rna_pred.obs['latent_time_mean'] = latent_time_mean
        adata_rna_pred.obs['latent_time_std'] = latent_time_std
        adata_rna_pred.obs['latent_time'] = latent_time_sorted

        adata_rna_pred.obsm['z0_mean'] = z0_mean
        adata_rna_pred.obsm['z_mean'] = z_mean
        adata_rna_pred.obsm['z_std'] = z_std
        adata_rna_pred.obsm['vz_mean'] = vz_mean
        adata_rna_pred.obsm['vz_std'] = vz_std
        adata_rna_pred.layers['x_pred_mean'] = x_pred_mean
        adata_rna_pred.layers['x_pred_std'] = x_pred_std
        adata_rna_pred.layers['vx_pred_mean'] = vx_pred_mean
        adata_rna_pred.layers['vx_pred_std'] = vx_pred_std

        return adata_rna_pred


    def calculate_jacobians(self, adata_rna_pred, dataloader, genes_of_interest, epsilon=1e-4):
        """
        Calculates the Jacobians of the model and adds them to the adata.

        Parameters
        ----------
        adata_rna_pred: RNA adata
        dataloader: Pytorch dataloader for all data
        genes_of_interest: genes for which to calculate the Jacobians
        epsilon: a small number used to calculate the Jacobians

        Returns
        -------
        adata_rna_pred: updated adata_rna containing the calculated Jacobian
        """

        batch_size = self.batch_size
        device = self.device
        n_cells = adata_rna_pred.shape[0]
        n_genes = adata_rna_pred.shape[1]

        genes = adata_rna_pred.var_names.values
        n_gi = len(genes_of_interest)
        genes_of_interest_idx = np.zeros(n_gi).astype(np.int32)
        for idx, g in enumerate(genes_of_interest):
            genes_of_interest_idx[idx] = np.where(genes==g)[0].item()

        J_dvx_dx = np.zeros([n_cells, n_gi, n_gi])

        with torch.no_grad():
            self.eval()
            for idx_batch, (x, vx, v_mask, w) in enumerate(dataloader):
                print(f'Batch {idx_batch+1}/{len(dataloader)}')
                x = x.to(device)
                x_pred, vx_pred, z0_mean, z, vz, latent_time, latent_time_std = self(x)

                for i, g in enumerate(genes_of_interest_idx):
                    e = np.zeros(n_genes)
                    e[g] = 1.
                    x_epsilon = x + epsilon * torch.tensor(e).type(torch.float32).to(device)
                    x_pred, vx_pred_epsilon, z0_mean, z, vz, latent_time, latent_time_std = self(x_epsilon)

                    J_dvx_dx[idx_batch*batch_size:idx_batch*batch_size+batch_size, :, i] = (1/epsilon) * (vx_pred_epsilon.detach().cpu().numpy()[:, genes_of_interest_idx] - vx_pred.detach().cpu().numpy()[:, genes_of_interest_idx])

        # save jacobians to adata
        adata_rna_pred.obsm['J_dvx_dx'] = J_dvx_dx
        adata_rna_pred.uns['Jacobians:genes_of_interest'] = genes_of_interest

        return adata_rna_pred


    def predict_perturbation(self, adata_rna_pred, dataloader, perturbed_genes):
        """
        Performs in-silico perturbations and adds them to the adata

        Parameters
        ----------
        adata_rna_pred: RNA adata
        dataloader: Pytorch dataloader for all data
        perturbed_genes: genes to perturb

        Returns
        -------
        adata_rna_pred: updated adata_rna containing the post-perturbation delta velocities 
        """

        batch_size = self.batch_size
        z_dim = self.zx_dim
        device = self.device
        n_cells = adata_rna_pred.shape[0]
        n_genes = adata_rna_pred.shape[1]
        n_pt_genes = len(perturbed_genes)

        vz = np.zeros([n_cells, z_dim])
        vx_pred = np.zeros([n_cells, n_genes])
        latent_time_mean = np.zeros(n_cells)

        vz_perturbed = np.zeros([n_cells, z_dim])
        vx_pred_perturbed = np.zeros([n_cells, n_genes])
        latent_time_mean_perturbed = np.zeros(n_cells)

        delta_vx = np.zeros([n_cells, n_genes, n_pt_genes])
        delta_vz = np.zeros([n_cells, z_dim, n_pt_genes])
        delta_latent_time = np.zeros([n_cells, n_pt_genes])

        with torch.no_grad():
            self.eval()
            for idx_batch, (x, vx, v_mask, w) in enumerate(dataloader):
                x = x.to(device)
                x_pred_, vx_pred_, z0_mean_, z_, vz_, latent_time_, latent_time_std_ = self(x)

                latent_time_mean[idx_batch*batch_size:idx_batch*batch_size+batch_size] = latent_time_.detach().cpu().numpy()
                vz[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vz_.detach().cpu().numpy()
                vx_pred[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vx_pred_.detach().cpu().numpy()

            for idx, g in enumerate(perturbed_genes):
                print(f'idx: {idx} / perturbed gene: {g}')
                gene_idx = np.where(adata_rna_pred.var_names==g)[0].item()
                x_gene_min = adata_rna_pred.X[:,gene_idx].min()

                for idx_batch, (x, vx, v_mask, w) in enumerate(dataloader):
                    x[:,gene_idx] = torch.tensor(x_gene_min, dtype=torch.float32)
                    x_pred_, vx_pred_, z0_mean_, z_, vz_, latent_time_, latent_time_std_ = self(x)

                    latent_time_mean_perturbed[idx_batch*batch_size:idx_batch*batch_size+batch_size] = latent_time_.detach().cpu().numpy()
                    vz_perturbed[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vz_.detach().cpu().numpy()
                    vx_pred_perturbed[idx_batch*batch_size:idx_batch*batch_size+batch_size, :] = vx_pred_.detach().cpu().numpy()

                # delta
                delta_vx[:,:,idx] = vx_pred_perturbed - vx_pred
                delta_vz[:,:,idx] = vz_perturbed - vz
                delta_latent_time[:,idx] = latent_time_mean_perturbed - latent_time_mean

        # save deltas to adata
        adata_rna_pred.obsm['delta_vx'] = delta_vx
        adata_rna_pred.obsm['delta_vz'] = delta_vz
        adata_rna_pred.obsm['delta_latent_time'] = delta_latent_time
        adata_rna_pred.uns['Perturbed_genes'] = perturbed_genes

        return adata_rna_pred