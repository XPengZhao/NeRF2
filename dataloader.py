# -*- coding: utf-8 -*-
"""dataset processing and loading
"""
import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm


def split_dataset(datadir, ratio=0.8):
    """random shuffle train/test set
    """
    spectrum_dir = os.path.join(datadir, 'spectrum')
    spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
    index = [x.split('.')[0] for x in spt_names]
    random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')



class Spectrum_dataset(Dataset):
    """spectrum dataset class
    """
    def __init__(self, datadir, indexdir) -> None:
        super().__init__()
        self.datadir = datadir
        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        self.gateway_pos_dir = os.path.join(datadir, 'gateway_info.yml')
        self.spectrum_dir = os.path.join(datadir, 'spectrum')
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])
        example_spt = imageio.imread(os.path.join(self.spectrum_dir, self.spt_names[0]))
        self.n_elevation, self.n_azimuth = example_spt.shape
        self.rays_per_spectrum = self.n_elevation * self.n_azimuth
        self.dataset_index = np.loadtxt(indexdir, dtype=str)
        self.nn_inputs, self.nn_labels = self.load_data()


    def __len__(self):
        return len(self.dataset_index) * self.rays_per_spectrum


    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]


    def load_data(self):
        """load data from datadir to memory for training

        Returns
        -------
        train_inputs : tensor. [n_samples, 9]. The inputs for training
                  ray_o, ray_d, tx_pos
        """
        ## NOTE! Each spectrum will cost 1.2 MB of memory. Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 9)), dtype=torch.float32)
        nn_labels = torch.tensor(np.zeros((len(self), 1)), dtype=torch.float32)

        ## Load gateway position and orientation
        with open(os.path.join(self.gateway_pos_dir)) as f:
            gateway_info = yaml.safe_load(f)
            gateway_pos = gateway_info['gateway1']['position']
            gateway_orientation = gateway_info['gateway1']['orientation']

        ## Load transmitter position
        tx_pos = pd.read_csv(self.tx_pos_dir).values
        tx_pos = torch.tensor(tx_pos, dtype=torch.float32)

        ## Load data, each spectrum contains 90x360 pixels(rays)
        for i, idx in tqdm(enumerate(self.dataset_index), total=len(self.dataset_index)):
            spectrum = imageio.imread(os.path.join(self.spectrum_dir, idx + '.png')) / 255
            spectrum = torch.tensor(spectrum, dtype=torch.float32).view(-1, 1)
            ray_o, ray_d = self.gen_rays_spectrum(gateway_pos, gateway_orientation)
            tx_pos_i = torch.tile(tx_pos[int(idx)-1], (self.rays_per_spectrum,)).reshape(-1,3)  # [n_rays, 3]
            nn_inputs[i * self.rays_per_spectrum: (i + 1) * self.rays_per_spectrum, :9] = \
                torch.cat([ray_o, ray_d, tx_pos_i], dim=1)
            nn_labels[i * self.rays_per_spectrum: (i + 1) * self.rays_per_spectrum, :] = spectrum

        return nn_inputs, nn_labels


    def gen_rays_spectrum(self, gateway_pos, gateway_orientation):
        """generate sample rays origin at gateway with resolution given by spectrum

        Parameters
        ----------
        azimuth : int. The number of azimuth angles
        elevation : int. The number of elevation angles

        Returns
        -------
        r_o : tensor. [n_rays, 3]. The origin of rays
        r_d : tensor. [n_rays, 3]. The direction of rays, unit vector
        """

        azimuth = torch.linspace(1, 360, self.n_azimuth) / 180 * np.pi
        elevation = torch.linspace(1, 90, self.n_elevation) / 180 * np.pi
        azimuth = torch.tile(azimuth, (self.n_elevation,))  # [1,2,3...360,1,2,3...360,...] pytorch 2.0
        elevation = torch.repeat_interleave(elevation, self.n_azimuth)  # [1,1,1,...,2,2,2,...,90,90,90,...]

        x = 1 * torch.cos(elevation) * torch.cos(azimuth) # [n_azi * n_ele], i.e., [n_rays]
        y = 1 * torch.cos(elevation) * torch.sin(azimuth)
        z = 1 * torch.sin(elevation)

        r_d = torch.stack([x, y, z], dim=0)  # [3, n_rays] 3D direction of rays in gateway coordinate
        R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
        r_d = R @ r_d  # [3, n_rays] 3D direction of rays in world coordinate
        gateway_pos = torch.tensor(gateway_pos, dtype=torch.float32)
        r_o = torch.tile(gateway_pos, (self.rays_per_spectrum,)).reshape(-1, 3)  # [n_rays, 3]

        return r_o, r_d.T


