import sys
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import torch
import os
from glob import glob
import json
import matplotlib.pyplot as plt


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (Tensor): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    window = window.to(x.device)
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False, pad_mode='constant')
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


def convert_equirect_to_camera_coord(depth_map, img_h, img_w):
    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w))
    theta_map = (theta + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
    sin_theta = torch.sin(theta_map)
    cos_theta = torch.cos(theta_map)
    sin_phi = torch.sin(phi_map)
    cos_phi = torch.cos(phi_map)
    # print("GGG ", depth_map.max(), depth_map.min())
    return torch.stack([depth_map * cos_phi * cos_theta, depth_map * cos_phi * sin_theta, -depth_map * sin_phi], dim=-1)


def get_3d_point_camera_coord(rotation_angle, listener_pos, point_3d):
    camera_matrix = None
    lis_x, lis_y, lis_z = listener_pos[0], listener_pos[1], listener_pos[2]
    # print(lis_x, lis_y, lis_z)
    if rotation_angle == 0:
        camera_matrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        camera_matrix[:3, 3] = np.array([-lis_x, -lis_y, -lis_z])
    elif rotation_angle == 90:
        camera_matrix = np.array([[0., 0., -1., 0.], [0., 1., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
        camera_matrix[:3, 3] = np.array([lis_z, -lis_y, -lis_x])
    elif rotation_angle == 180:
        camera_matrix = np.array([[-1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., -1., 0.], [0., 0., 0., 1.]])
        camera_matrix[:3, 3] = np.array([lis_x, -lis_y, lis_z])
    elif rotation_angle == 270:
        camera_matrix = np.array([[0., 0., 1., 0.], [0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 0., 1.]])
        camera_matrix[:3, 3] = np.array([-lis_z, -lis_y, lis_x])
    # point_3d[1] += 1.5528907
    point_4d = np.append(point_3d, 1.0)
    camera_coord_point = camera_matrix @ point_4d
    return camera_coord_point[:3]



class HAA_Classroom_RIR_Dataset(Dataset):

    def __init__(self, split="train", max_len=9600, num_shot = 4,
    pano_depth_path="/path/to/new_processed_class_room/depth_src/0.npy",
     ir_path="/path/to/new_processed_class_room/single_channel_ir/", 
     metadata_path="/path/to/new_processed_class_room/xyzs.npy"):
        self.split = split
        self.max_len = max_len
        self.num_shot = num_shot
        self.pano_depth_path = pano_depth_path
        self.ir_path = ir_path
        self.metadata_path = metadata_path

        self.train_indices = np.arange(12)*(57)
        self.train_tgt_indices = np.arange(0, 315) * 2
        to_exclude = list(self.train_indices) + list(self.train_tgt_indices)
        self.test_indices = np.array([i for i in range(0, 630) if i not in set(to_exclude)])

        self.source_locs = np.load(self.metadata_path)
        self.rec_locs = np.array([3.5838, 5.7230, 1.2294])

        self.pano_depth = np.load(pano_depth_path)
        self.depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(self.pano_depth), 256, 512)
        self.file_list = glob(ir_path + "*.wav")
        if self.split == "test":
            self.file_list = [fp for fp in self.file_list if int(os.path.basename(fp)[:-4]) in set(list(self.test_indices))]
        else:
            self.file_list = [fp for fp in self.file_list if int(os.path.basename(fp)[:-4]) in set(list(self.train_indices))]
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        ir_file_path = self.file_list[idx]
        src_index = int(os.path.basename(ir_file_path)[:-4])
        listener_pos = self.rec_locs.copy()
        source_pos = self.source_locs[src_index].copy()
        rand_rotation = 0.0
        proj_source_pos = get_3d_point_camera_coord(rand_rotation, listener_pos, source_pos)
        proj_listener_pos = np.array([0., 0., 0.])
        
        depth_coord = self.depth_coord 
        
        tgt_wav, rate = torchaudio.load(ir_file_path)
        assert rate == 22050, "IR sampling rate must be 22050!"
        if tgt_wav.shape[1] < self.max_len:
            tgt_wav = torch.cat([tgt_wav, torch.zeros(tgt_wav.shape[0], self.max_len - tgt_wav.shape[1])], dim=1)
        else:
            tgt_wav = tgt_wav[:, :self.max_len]
        
        num_ref_sources = self.num_shot
        sel_other_src_indices = np.random.choice([i for i in list(self.train_indices) if i != src_index], num_ref_sources, replace=False)
        all_ref_src_pos_ori = self.source_locs[sel_other_src_indices,:].copy()
        all_ref_src_pos = []
        for i in range(all_ref_src_pos_ori.shape[0]):
            proj_src_loc_ref = get_3d_point_camera_coord(0., listener_pos, all_ref_src_pos_ori[i])
            all_ref_src_pos.append(torch.Tensor(proj_src_loc_ref).float())
        all_ref_src_pos = torch.vstack(all_ref_src_pos)
        all_ref_irs = []
        for idx in sel_other_src_indices:
            ref_wav, rate = torchaudio.load(os.path.join(self.ir_path, f"{idx}.wav"))
            assert rate == 22050, "IR sampling rate must be 22050!"
            if ref_wav.shape[1] < self.max_len:
                ref_wav = torch.cat([ref_wav, torch.zeros(ref_wav.shape[0], self.max_len - ref_wav.shape[1])], dim=1)
            else:
                ref_wav = ref_wav[:, :self.max_len]
            all_ref_irs.append(ref_wav)
        all_ref_irs = torch.cat(all_ref_irs, dim=0)
        return  torch.Tensor(proj_listener_pos).float(), torch.Tensor(proj_source_pos).float(), torch.Tensor(depth_coord).permute(2, 0, 1).float(),  tgt_wav, all_ref_irs, all_ref_src_pos #src_index, mean, std, norm_tgt_spec tgt_spec,

    
        

if __name__ == "__main__":
    dataset = Position_to_IR_Dataset(num_shot = 8)
    print(len(dataset))
    for i in range(len(dataset)):
        proj_listener_pos, proj_source_pos, depth_coord, tgt_wav, all_ref_irs, all_ref_src_pos = dataset.__getitem__(i)
        print(proj_listener_pos, proj_source_pos)
        print(depth_coord[:, 128, 128])
