# -*- coding: utf-8 -*-
"""code for ray marching and signal rendering
"""
import torch
import numpy as np
import torch.nn.functional as F

class Renderer():

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """

        ## Rendering parameters
        self.network_fn = networks_fn
        self.n_samples = kwargs['n_samples']
        self.near = kwargs['near']
        self.far = kwargs['far']



    def render_ss(self, tx, rays_o, rays_d):
        """render the signal strength of each ray

        Parameters
        ----------
        tx: tensor. [batchsize, 3]. The position of the transmitter
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 3]. The direction of rays
        """

        # sample points along rays
        batch_size = rays_o.shape[0]
        near, far = torch.full((batch_size, 1), self.near), torch.full((batch_size, 1), self.far)
        t_vals = torch.linspace(0., 1., steps=self.n_samples) * (far - near) + near  # scale t with near and far
        t_vals = t_vals.to(rays_o.device)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * t_vals[...,:,None] # p = o + td, [batch, n_samples, 3]

        # Expand views and tx to match the shape of pts
        view = rays_d[:, None].expand(pts.shape)
        tx = tx[:, None].expand(pts.shape)

        # Run network and compute outputs
        raw = self.network_fn(pts, view, tx)    # [batchsize, n_samples, 4]
        receive_ss = self.raw2outputs(raw, t_vals, rays_d)  # [batchsize]

        return receive_ss



    def raw2outputs(self, raw, r_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values. (core part)

        Parameters
        ----------
        raw : [batchsize, n_samples, 4]. Prediction from model.
        r_vals : [batchsize, n_samples]. Integration distance.
        rays_d : [batchsize, 3]. Direction of each ray

        Return:
        ----------
        receive_signal : [batchsize]. abs(singal of each ray)
        """

        raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
        # raw2phase = lambda raw, dists: torch.exp(1j*raw*dists)
        raw2phase = lambda raw, dists: raw*dists

        dists = r_vals[...,1:] - r_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        att_a, att_p, s_a, s_p = raw[...,0], raw[...,1], raw[...,2], raw[...,3]    # [N_rays, N_samples]
        att_p, s_p = torch.sigmoid(att_p)*np.pi*2, torch.sigmoid(s_p)*np.pi*2
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))
        # att_a, s_a = torch.sigmoid(att_a), torch.sigmoid(s_a)

        alpha = raw2alpha(att_a, dists)  # [N_rays, N_samples]
        phase = raw2phase(att_p, dists)

        att_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        path = torch.cat([r_vals[...,1:], torch.Tensor([1e10]).cuda().expand(r_vals[...,:1].shape)], -1)
        path_loss = 0.025 / path
        # phase_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), phase], -1), -1)[:, :-1]
        phase_i = torch.cumsum(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), phase], -1), -1)[:, :-1]
        phase_i = torch.exp(1j*phase_i)    # [N_rays, N_samples]


        receive_signal = torch.sum(s_a*torch.exp(1j*s_p)*att_i*phase_i*path_loss, -1)  # [N_rays]
        receive_signal = abs(receive_signal)

        return receive_signal