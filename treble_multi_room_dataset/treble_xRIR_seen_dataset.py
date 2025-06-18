import sys
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import torch
from utils.spec_utils import stft
import os, pickle
from glob import glob
import json
import matplotlib.pyplot as plt


BASE_DATA_PATH = "/pscratch/sd/x/xiuliu/room_rir/preprocess_treble_dense/"




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


def compute_energy_db(h):
    h = np.array(h)
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]
    return  energy_db


class xRIR_Dataset(Dataset):

    def __init__(self, split="train", max_len=9600, num_shot = 4,
    pano_depth_path="depth_map",
     ir_path="single_channel_ir", 
     metadata_path="metadata"):
        self.split = split
        self.max_len = max_len
        self.num_shot = num_shot
        self.pano_depth_path = os.path.join(BASE_DATA_PATH, pano_depth_path)
        self.ir_path = os.path.join(BASE_DATA_PATH, ir_path)
        self.metadata_path = os.path.join(BASE_DATA_PATH, metadata_path)
        self.scene_categories = ["Apartments", "Bathrooms", "Cafe", "LivingRoomsWithHallway", "Office",
                                 "Auditorium", "Bedrooms", "ListeningRoom", "MeetingRoom", "Restaurants"]
        self.scene_dir = [os.path.join(self.ir_path, scene) for scene in self.scene_categories]
        self.all_scenes = []
        for cur_dir in self.scene_dir:
            cur_scene_dirs = [os.path.join(cur_dir, fn) for fn in os.listdir(cur_dir)]
            self.all_scenes.extend(cur_scene_dirs)
                    
        self.all_scene_files = []
        self.test_scene_files = []
        with open("treble_multi_room_dataset/seen_test_split.pkl", "rb") as fin:
            test_scene_files = pickle.load(fin)
        for fp in test_scene_files:
            self.test_scene_files.append(os.path.join(self.ir_path, fp))
        for fp in self.all_scenes:
            self.all_scene_files.extend(glob(fp + "/*.wav"))
        self.train_scene_files = list(set(self.all_scene_files).difference(self.test_scene_files))

        print(len(self.test_scene_files), len(self.train_scene_files))

        assert split in ["train", "test"], "Split must be either train or test!"
        if self.split == "train":
            self.file_list = self.train_scene_files
        else:
            self.file_list = self.test_scene_files
        # print(self.file_list)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        ir_file_path = self.file_list[idx]
        ir_file_name = os.path.basename(ir_file_path).split("_hybrid_IR")[0]
        scene_name = ir_file_path.split("/")[-3]
        scene_id = ir_file_path.split("/")[-2]
        receiver_idx, source_idx = int(ir_file_name.split("_")[1][1:]), int(ir_file_name.split("_")[0][1:])
        source_pos, listener_pos = self.get_receiver_source_location(ir_file_path)
        
        rotation = 0
        proj_source_pos = get_3d_point_camera_coord(rotation, listener_pos, source_pos)
        proj_listener_pos = np.array([0., 0., 0.])
        pano_depth = np.load(os.path.join(self.pano_depth_path, scene_name, scene_id, f"{receiver_idx}.npy"))
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth), 256, 512)
        tgt_wav, rate = torchaudio.load(ir_file_path)
        assert rate == 22050, "IR sampling rate must be 22050!"
        if tgt_wav.shape[1] < self.max_len:
            tgt_wav = torch.cat([tgt_wav, torch.zeros(tgt_wav.shape[0], self.max_len - tgt_wav.shape[1])], dim=1)
        else:
            tgt_wav = tgt_wav[:, :self.max_len]
        
        all_ref_irs, all_ref_src_pos = self.get_ir_and_location_for_other_sources(ir_file_path, num_ref_sources=self.num_shot)


        return torch.Tensor(proj_listener_pos).float(), torch.Tensor(proj_source_pos).float(), torch.Tensor(depth_coord).permute(2, 0, 1).float(),  tgt_wav, all_ref_irs, all_ref_src_pos #mean, std, norm_tgt_spec tgt_spec,

    def get_receiver_source_location(self, ir_file_path):
        scene_name = ir_file_path.split("/")[-3]
        scene_id = ir_file_path.split("/")[-2]
        ir_file_name = ir_file_path.split("/")[-1]
        src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
        json_file_name = "S00" + str(src_node) + "_R00" + str(rec_node) + ".json"
        metadata_file_path = os.path.join(self.metadata_path, scene_name, scene_id, json_file_name)
        with open(metadata_file_path, "r") as fin:
            meta_info = json.load(fin)
        src_loc = meta_info["src_loc"]
        rec_loc = meta_info["rec_loc"]
        return src_loc, rec_loc
    
    def get_ir_and_location_for_other_sources(self, ir_file_path, num_ref_sources):
        dir_name = os.path.dirname(ir_file_path)
        ir_file_name = ir_file_path.split("/")[-1]
        src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
        all_src_node = set([int(fn.split("_")[0][1:]) for fn in os.listdir(dir_name)])
        remain_src_node = list(all_src_node.difference(set([src_node])))
        valid_other_src_ir_paths = []
        for node in remain_src_node:
            rec_n = ir_file_name.split("_")[1]
            src_n = f"S00{node}"
            other_src_ir_path = os.path.join(dir_name, f"{src_n}_{rec_n}_hybrid_IR.wav")
            if os.path.exists(other_src_ir_path):
                valid_other_src_ir_paths.append(other_src_ir_path)
        try:
            select_other_src_ir_paths = np.random.choice(valid_other_src_ir_paths, num_ref_sources, replace=False)
        except Exception as e:
            select_other_src_ir_paths = np.random.choice(valid_other_src_ir_paths, num_ref_sources, replace=True)
        all_ref_irs = []
        all_ref_src_pos = []
      
        for fp in select_other_src_ir_paths:
            ref_wav, rate = torchaudio.load(fp)
            assert rate == 22050, "IR sampling rate must be 22050!"
            if ref_wav.shape[1] < self.max_len:
                ref_wav = torch.cat([ref_wav, torch.zeros(ref_wav.shape[0], self.max_len - ref_wav.shape[1])], dim=1)
            else:
                ref_wav = ref_wav[:, :self.max_len]
            all_ref_irs.append(ref_wav)

            src_loc, rec_loc = self.get_receiver_source_location(fp)
            
            proj_src_loc = get_3d_point_camera_coord(0, rec_loc, src_loc)
            
            all_ref_src_pos.append(torch.Tensor(proj_src_loc).float())
        all_ref_irs = torch.cat(all_ref_irs, dim=0)
        all_ref_src_pos = torch.vstack(all_ref_src_pos)
        return all_ref_irs, all_ref_src_pos
        
        

if __name__ == "__main__":
    dataset = xRIR_Dataset(num_shot = 8, split="train")
    print(len(dataset))
    exit(0)
    for i in range(len(dataset)):
        proj_listener_pos, proj_source_pos, depth_coord, tgt_wav, all_ref_irs, all_ref_src_pos = dataset.__getitem__(i) #tgt_spec, 
        # print(all_ref_irs.shape)
        # print(all_ref_src_pos)
        # print("x"*30)
        