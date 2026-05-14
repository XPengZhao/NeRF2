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
csi2snr = lambda x, y: -10 * torch.log10(
    torch.norm(x - y, dim=(1, 2)) ** 2 /
    torch.norm(y, dim=(1, 2)) ** 2
)




class Embedder():
    """positional encoding
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        d = self.kwargs['input_dims']    # input dimension of gamma

        max_freq = self.kwargs['max_freq_log2']    # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']         # L

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  #2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        self.freq_bands = freq_bands
        self.include_input = self.kwargs['include_input']
        self.out_dim = d * (int(self.include_input) + 2 * N_freqs)

    def embed(self, inputs):
        """return: gamma(input)
        """
        freq_bands = self.freq_bands.to(device=inputs.device, dtype=inputs.dtype)
        scaled = inputs[..., None, :] * freq_bands.view(*([1] * (inputs.dim() - 1)), -1, 1)
        encoded = torch.stack((torch.sin(scaled), torch.cos(scaled)), dim=-2).flatten(-3)
        if self.include_input:
            encoded = torch.cat([inputs, encoded], dim=-1)
        return encoded




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
                'include_input' : True,
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

    def __init__(self, D=8, W=256, skips=[4],
                 input_dims={'pts':3, 'view':3, 'tx':3},
                 multires = {'pts':10, 'view':10, 'tx':10},
                 is_embeded={'pts':True, 'view':True, 'tx':False},
                 attn_output_dims=2, sig_output_dims=2):
        """NeRF2 model

        Parameters
        ----------
        D : int, hidden layer number, default by 8
        W : int, Dimension per hidden layer, default by 256
        skip : list, skip layer index
        input_dims: dict, input dimensions
        multires: dict, log2 of max freq for position, view, and tx position positional encoding, i.e., (L-1)
        is_embeded : dict, whether to use positional encoding
        attn_output_dims : int, output dimension of attenuation
        sig_output_dims : int, output dimension of signal
        """

        super().__init__()
        self.skips = skips

        # set positional encoding function
        self.embed_pts_fn, input_pts_dim = get_embedder(multires['pts'], is_embeded['pts'], input_dims['pts'])
        self.embed_view_fn, input_view_dim = get_embedder(multires['view'], is_embeded['view'], input_dims['view'])
        self.embed_tx_fn, input_tx_dim = get_embedder(multires['tx'], is_embeded['tx'], input_dims['tx'])
        self.input_view_dim = input_view_dim
        self.input_tx_dim = input_tx_dim

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
        self.attenuation_output = nn.Linear(W, attn_output_dims)
        self.feature_layer = nn.Linear(W, W)
        self.signal_output = nn.Linear(W//2, sig_output_dims)

    @staticmethod
    def _expand_to_points(encoded, point_shape):
        while encoded.dim() - 1 < len(point_shape):
            encoded = encoded.unsqueeze(-2)
        return encoded.expand(*point_shape, encoded.shape[-1])

    def _first_signal_layer(self, feature, view, tx, point_shape):
        layer = self.signal_linears[0]
        feature_dim = feature.shape[-1]
        feature_w, view_w, tx_w = torch.split(
            layer.weight,
            [feature_dim, self.input_view_dim, self.input_tx_dim],
            dim=1
        )

        x = F.linear(feature, feature_w, layer.bias).view(*point_shape, -1)
        view = self._expand_to_points(F.linear(view, view_w), point_shape)
        tx = self._expand_to_points(F.linear(tx, tx_w), point_shape)
        return (x + view + tx).reshape(-1, x.shape[-1])


    def forward(self, pts, view, tx):
        """forward function of the model

        Parameters
        ----------
        pts: [batchsize, n_samples, 3], position of voxels
        view: [batchsize, 3] or [batchsize, n_samples, 3], view direction
        tx: [batchsize, 3] or [batchsize, n_samples, 3], transmitter/uplink feature

        Returns
        ----------
        outputs: [batchsize, n_samples, 4].   attn_amp, attn_phase, signal_amp, signal_phase
        """

        point_shape = pts.shape[:-1]

        # position encoding
        pts = self.embed_pts_fn(pts).contiguous()
        view = self.embed_view_fn(view)
        tx = self.embed_tx_fn(tx)
        pts = pts.reshape(-1, pts.shape[-1])

        x = pts
        for i, layer in enumerate(self.attenuation_linears):
            x = F.relu(layer(x))
            if i in self.skips:
                x = torch.cat([pts, x], -1)

        attn = self.attenuation_output(x)    # (batch_size, 2)
        feature = self.feature_layer(x)

        x = F.relu(self._first_signal_layer(feature, view, tx, point_shape))
        for layer in self.signal_linears[1:]:
            x = F.relu(layer(x))
        signal = self.signal_output(x)    #[batchsize, n_samples, 2]

        outputs = torch.cat([attn, signal], -1).contiguous()    # [batchsize, n_samples, 4]
        return outputs.view(*point_shape, outputs.shape[-1])
