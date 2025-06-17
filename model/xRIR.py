
import sys
from utils.spec_utils import stft
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import auraloss
import math
import numpy as np
from model.simple_vit import SimpleViT
import torchvision.models as models


def compute_energy_db(h):
    h = np.array(h)
    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]
    return  energy_db


def apply_delay(signal, delay_tensor):
    """
    Apply integer delay to each sample in the batch with zero-padding.
    
    Args:
    - signal: Tensor of shape [batch_size, sequence_length, channels] or [batch_size, sequence_length].
    - delay_tensor: Tensor of shape [batch_size] with integer delay values for each sample.
    
    Returns:
    - Delayed signal with the same shape as input signal.
    """
    batch_size, sequence_length = signal.shape[:2]
    
    # Output tensor initialized with zeros (same shape as input signal)
    delayed_signal = torch.zeros_like(signal).cuda()
    
    for i in range(batch_size):
        delay = delay_tensor[i].item()  # Get the delay value for this batch item
        
        if delay > 0:
            # Positive delay: shift right and pad with zeros at the beginning
            delayed_signal[i, delay:] = signal[i, :-delay]
        elif delay < 0:
            # Negative delay: shift left and pad with zeros at the end
            delayed_signal[i, :delay] = signal[i, -delay:]
        else:
            # No delay, just copy the signal
            delayed_signal[i] = signal[i]
    
    return delayed_signal



class AudioEnc(nn.Module):
    def __init__(self, log_instead_of_log1p_in_logspace=True,
                 log_eps=1.0e-8, ):
        """
        ResNet-18.
        Takes in observations (binaural IR magnitude spectrograms) and produces an acoustic embedding

        :param log_instead_of_log1p_in_logspace: compute log of magnitude spect. instead of log(1 + ...)
        :param log_eps: epsilon to be used to compute log for numerical stability
        """
        super().__init__()

        self._log_instead_of_log1p_in_logspace = log_instead_of_log1p_in_logspace
        self._log_eps = log_eps

        self._n_input = 1

        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc_backup = self.cnn.fc
        self.cnn.fc = nn.Sequential()

        self.cnn.conv1 = nn.Conv2d(self._n_input,
                                   self.cnn.conv1.out_channels,
                                   kernel_size=self.cnn.conv1.kernel_size,
                                   stride=self.cnn.conv1.stride,
                                   padding=self.cnn.conv1.padding,
                                   bias=False)

        nn.init.kaiming_normal_(
            self.cnn.conv1.weight, mode="fan_out", nonlinearity="relu",
        )

    @property
    def n_out_feats(self):
        """
        get number of audio encoder features
        :return: number of audio encoder features
        """
        # resnet-18
        return 512

    def forward(self, audio_spect,):
        """
        does forward pass in audio encoder
        :param audio_spect: audio spectrogram observations
        :return: acoustic/audio features
        """
        cnn_input = []
        if self._log_instead_of_log1p_in_logspace:
            audio_spect_observations = torch.log(audio_spect + self._log_eps)
        else:
            audio_spect_observations = torch.log1p(audio_spect) 
        cnn_input.append(audio_spect_observations)
        cnn_input = torch.cat(cnn_input, dim=1)
        return self.cnn(cnn_input)




class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)


class basic_project2(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(basic_project2, self).__init__()
        self.proj = nn.Linear(input_ch, output_ch, bias=True)
    def forward(self, x):
        return self.proj(x)



class xRIR(nn.Module):
    def __init__(self, condition_dim, num_channels, n_bins=310, dim=512, intermediate_ch=256, input_dim=1, latent_dim=512,
       all_block=6, image_size = (256, 512), patch_size = (16, 32), depth=12, mlp_dim=512, heads=8, time_input_ch=(13 * 2 + 1) * 7
    ):
        super(xRIR, self).__init__()
        self.num_channels = num_channels
        self.t_bins = n_bins

        self.source_network = SimpleViT(image_size, patch_size, dim, depth, heads, mlp_dim, channels=3)
        self.src_proj = basic_project2(dim, intermediate_ch)

        self.dist_embedder = embedding_module_log(num_freqs=10, max_freq=7, ch_dim=1,)
        # self.patch_embedder = embedding_module_log(num_freqs=10, max_freq=7, ch_dim=2,)
        src_lis_input_ch= 21 * 3
        self.src_coord_proj = basic_project2(src_lis_input_ch, intermediate_ch)
        self.source_proj = basic_project2(21 * 2, intermediate_ch)
        self.lin_proj_0 = basic_project2(intermediate_ch, 1)
        self.lin_proj_1 = basic_project2(intermediate_ch * 3, intermediate_ch) # num_patch x 1
        self.lin_proj_2 = basic_project2(intermediate_ch * 5, intermediate_ch)
        self.audio_enc = AudioEnc()

        self.t_bins = n_bins
        self.times = 2*(torch.arange(0, self.t_bins))/self.t_bins - 1.0
        self.time_embedder = embedding_module_log(num_freqs=10, ch_dim=2)
        self.times = self.times.unsqueeze(1)
        self.time_proj = basic_project2(21, intermediate_ch)



    
    def forward(self, depth_coord, x, src_loc, ref_ir_locs, tgt_wav):
        
        x = self.shift_and_align(x, src_loc, ref_ir_locs)
        times = self.times
        time_embed = self.time_embedder(times.unsqueeze(0).to(ref_ir_locs.device)).repeat(ref_ir_locs.shape[0], 1, 1)  # B,T,21
        time_out = self.time_proj(time_embed) # B T C

        src_coord_encoding = (src_loc[:, :, None, None] - depth_coord) / 5.

        source_out = self.src_proj(self.source_network(src_coord_encoding)) # B x K x C
        source_out = self.lin_proj_0(source_out.permute(0, 2, 1)).squeeze(-1).unsqueeze(1) # B x 1 x C

        rec_coord_encoding = (-depth_coord) / 5. #self.dist_embedder(
        receiver_out = self.src_proj(self.source_network(rec_coord_encoding)) # B x K x C
        receiver_out = self.lin_proj_0(receiver_out.permute(0, 2, 1)).squeeze(-1).unsqueeze(1)

        ref_geo_feat_list = []

        for i in range(ref_ir_locs.shape[1]):
            ref_src_coord_encoding =  (ref_ir_locs[:, i, :, None, None] - depth_coord) / 5. #self.dist_embedder(
            ref_geo_feat = self.src_proj(self.source_network(ref_src_coord_encoding))
            ref_geo_feat = self.lin_proj_0(ref_geo_feat.permute(0, 2, 1)).squeeze(-1).unsqueeze(1) #b 1 C
            ref_geo_feat_list.append(ref_geo_feat)#.unsqueeze(1)
        ref_geo_feats = torch.cat(ref_geo_feat_list, dim=1) # BxNxC B x N x K x C

        fuse_geo_feats = torch.cat([receiver_out, source_out], dim=-1)
        fuse_ref_geo_feats = torch.cat([receiver_out.repeat(1, ref_geo_feats.shape[1], 1), ref_geo_feats], dim=-1) # B N 2C

        ref_src_feats = []

        for i in range(ref_ir_locs.shape[1]):
            ref_src_feat = self.src_coord_proj(self.dist_embedder((ref_ir_locs[:, i:(i+1)]) / 5.).view(ref_ir_locs.shape[0], -1)).unsqueeze(1)
            ref_src_feats.append(ref_src_feat)
        ref_src_feats = torch.cat(ref_src_feats, dim=1) # B N C
        src_feats = self.src_coord_proj(self.dist_embedder(src_loc.unsqueeze(1) / 5.0).view(src_loc.shape[0], -1)).unsqueeze(1) # B 1 C
        
        fuse_geo_feats = torch.cat([src_feats, fuse_geo_feats], dim=-1) # B 1 3*C
        fuse_ref_geo_feats = torch.cat([ref_src_feats, fuse_ref_geo_feats], dim=-1) # B N 3*C

        a_feats_all = []
        specs_all = []
        log_specs_all = []
        for i in range(x.shape[1]):
            spec_i = self.convert_ir_to_spec(x[:, i:(i+1)])
            specs_all.append(spec_i)
            log_specs_all.append(torch.log(spec_i + 1e-8))
            a_feats_i = self.audio_enc(spec_i)
            a_feats_all.append(a_feats_i.unsqueeze(1))
        a_feats_all = torch.cat(a_feats_all, dim=1) # B N 2*C
        specs_all = torch.cat(specs_all, dim=1) # B N 63 T
        log_specs_all = torch.cat(log_specs_all, dim=1)
        fuse_ref_feats = self.lin_proj_2(torch.cat((fuse_ref_geo_feats, a_feats_all), dim=-1)) #B N C
        fuse_tgt_feats = self.lin_proj_1(fuse_geo_feats) # B 1 C

        fuse_feats = F.softmax(fuse_ref_feats @ fuse_tgt_feats.permute(0, 2, 1) / fuse_tgt_feats.shape[-1], dim=1) * fuse_ref_feats # B N C
        weights = fuse_feats @ time_out.permute(0, 2, 1) / fuse_feats.shape[-1] # B N T
        out_log_spec = torch.sum(log_specs_all * weights.unsqueeze(2), dim=1).unsqueeze(1).permute(0, 2, 3, 1)
        tgt_spec = self.convert_ir_to_spec(tgt_wav)

        return out_log_spec, tgt_spec

    def convert_ir_to_spec(self, ir):
        tgt_spec = []
        for i in range(ir.shape[0]):
            tgt_spec.append(stft(
                ir[i],
                fft_size=124,
                hop_size=31,
                win_length=62,
                window=torch.hann_window(62)
            ).permute(0, 2, 1).unsqueeze(0))
        return torch.vstack(tgt_spec)


    def shift_and_align(self, x, src_loc, ref_ir_locs):
        channel_outputs = []
        dist_result = torch.linalg.norm(src_loc.unsqueeze(1) - ref_ir_locs, dim=-1)
        dist_src = torch.linalg.norm(src_loc, dim=1).unsqueeze(1)
        dist_ref = torch.linalg.norm(ref_ir_locs, dim=-1)
        direct_energy_ratio = dist_ref / (dist_src + 1e-7)
        delay_unit = torch.round((dist_src - dist_ref) / 343. * 22050).int()

        for i in range(x.shape[1]):
            cur_delayed_x = apply_delay(x[:, i, :], delay_unit[:, i]).unsqueeze(1)
            x_channel = cur_delayed_x * direct_energy_ratio[:, i:(i+1)].unsqueeze(2)
            channel_outputs.append(x_channel)
        return torch.cat(channel_outputs, dim=1)