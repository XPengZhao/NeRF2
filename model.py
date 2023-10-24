# -*- coding: utf-8 -*-
"""NeRF2 NN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2me = lambda x, y : torch.mean(abs(x - y))
sig2mse = lambda x, y : torch.mean((x - y) ** 2)

class Embedder():
    """positional encoding
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']    # input dimension of gamma
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']    # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']         # L


        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  #2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)




def get_embedder(multires, is_embeded=True, input_dims=3):
    """get positional encoding function

    Parameters
    ----------
    multires : log2 of max freq for positional encoding, i.e., (L-1)
    i : set 1 for default positional encoding, 0 for none
    input_dims : input dimension of gamma


    Returns
    -------
        embedding function; output_dims
    """
    if is_embeded == False:
        return nn.Identity(), input_dims

    embed_kwargs = {
                'include_input' : False,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim



class NeRF2(nn.Module):

    def __init__(self, D=8, W=256, multires_pts=10, multires_view=10, multires_tx=10, skips=[4], is_embeded=True):
        """NeRF2 model

        Parameters
        ----------
        D : int, hidden layer number, default by 8
        W : int, Dimension per hidden layer, default by 256
        multires_pts : int, the layer number of positional encoding for position in the scene
        multires_view : int, the layer number of positional encoding for view direction
        multires_tx : int, the layer number of positional encoding for transmitter position
        skip : list, skip layer index
        is_embeded : bool, whether to use positional encoding
        """

        super().__init__()
        self.skips = skips

        # set positional encoding function
        self.embed_pts_fn, input_pts_dim = get_embedder(multires_pts, is_embeded)
        self.embed_view_fn, input_view_dim = get_embedder(multires_view, is_embeded)
        self.embed_tx_fn, input_tx_dim = get_embedder(multires_tx, is_embeded)

        ## attenuation network
        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(input_pts_dim, W)] +
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_pts_dim, W)
             for i in range(D - 1)]
        )

        ## signal network
        self.signal_linears = nn.ModuleList(
            [nn.Linear(input_view_dim + input_tx_dim + W, W)] +
            [nn.Linear(W, W//2)]
        )

        ## output head, 2 for amplitude and phase
        self.attenuation_output = nn.Linear(W, 2)
        self.feature_layer = nn.Linear(W, W)
        self.signal_output = nn.Linear(W//2, 2)


    def forward(self, pts, view, tx):
        """forward function of the model

        Parameters
        ----------
        pts: [batchsize, n_samples, 3], position of voxels
        view: [batchsize, n_samples, 3], view direction
        tx: [batchsize, n_samples, 3], position of transmitter

        Returns
        ----------
        outputs: [batchsize, n_samples, 4].   attn_amp, attn_phase, signal_amp, signal_phase
        """

        # position encoding
        pts = self.embed_pts_fn(pts).contiguous()
        view = self.embed_view_fn(view).contiguous()
        tx = self.embed_tx_fn(tx).contiguous()
        shape = list(pts.shape)
        pts = pts.view(-1, shape[-1])
        view = view.view(-1, shape[-1])
        tx = tx.view(-1, shape[-1])

        x = pts
        for i, layer in enumerate(self.attenuation_linears):
            x = F.relu(layer(x))
            if i in self.skips:
                x = torch.cat([pts, x], -1)

        attn = self.attenuation_output(x)    # (batch_size, 2)
        feature = self.feature_layer(x)
        x = torch.cat([feature, view, tx], -1)

        for i, layer in enumerate(self.signal_linears):
            x = F.relu(layer(x))
        signal = self.signal_output(x)    #[batchsize, n_samples, 2]

        outputs = torch.cat([attn, signal], -1).contiguous()    # [batchsize, n_samples, 4]
        return outputs.view(shape[:-1]+[4])
