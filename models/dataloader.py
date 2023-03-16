import os
import pickle
import glob
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import pytorch_lightning as pl

def get_affine_matrix_quat(x, y, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])

class NavSet(Dataset):
    """ Dataset object representing the data from a single rosbag
    """

    def __init__(self, 
                 save_data_path: str, 
                 rosbag_path: str,
                 lidar_img_size = 240,
                 rgb_img_size = 240, 
                 rob_pose_len=32,
                 max_obj=15,
                 mot_pose_len=32) -> None:
        """ initialize a NavSet object,
            save path to data but do not load into RAM
        Args:
            save_data_path (str): path to the data pulled from a single rosbag
            rosbag_path (str): 
        """
        super().__init__()

        # save paths to lidar, rbg_img, pose data
        self.pose_path = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_pose.pkl'))
        self.lidar_dir = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_lidar_bev'))
        self.img_dir = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_rgb_img'))
        self.mot_dir = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_mot.pkl'))

        self.pose_data_points = pickle.load(open(self.pose_path, 'rb'))
        self.mot_data = pickle.load(open(self.mot_dir, 'rb'))

        self.max_obj = max_obj
        self.mot_pose_len = mot_pose_len
        self.rob_pose_len = rob_pose_len

        self.lidar_transforms = transforms.Compose([
            transforms.Resize((lidar_img_size,lidar_img_size)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])

        self.rgb_transforms = transforms.Compose([
            transforms.Resize((rgb_img_size,rgb_img_size)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
        ])

        self.length = max(0,len(os.listdir(self.lidar_dir)) - 2*mot_pose_len)

    def __len__(self) -> int:
        """ return the length of the of dataset
        """
        return self.length

    def __getitem__(self, index):
        rgb_img = Image.open(os.path.join(self.img_dir, f'{index}.png'))
        rgb_img = self.rgb_transforms(rgb_img)

        lidar_img = Image.open(os.path.join(self.lidar_dir, f'{index}.png'))
        lidar_img = self.lidar_transforms(lidar_img)

        curr_pose = self.pose_data_points['pose_sync'][index]
        curr_pose_mat_inv = np.linalg.pinv(get_affine_matrix_quat(curr_pose[0], curr_pose[1], curr_pose[2]))

        #goal_points = np.ones((3,len(self.pose_data_points['pose_future'][index])), dtype=np.float32)
        

        gt_pose = []
        for i in range(1,self.rob_pose_len+1):
            if i % 5 == 0:
                gt_pose.append(self.pose_data_points['pose_sync'][index+i])
        
        goal_points = np.ones((3,len(gt_pose)), dtype=np.float32)

        for i, goal_pose in enumerate(gt_pose):  #(self.pose_data_points['pose_sync'][index+1:index+1+self.rob_pose_len]):
            goal_points[0,i] = goal_pose[0]
            goal_points[1,i] = goal_pose[1]
        
        goal_points =  np.transpose(np.matmul(curr_pose_mat_inv, goal_points)[:-1,:])
        goal_tensor = torch.from_numpy(goal_points).to(torch.float32)
        #goal_tensor = torch.ones((20,2))
        #mot_data_points = pickle.load(open(os.path.join(self.mot_dir, f'{index}.pkl'), 'rb'))
        mot_data_points = self.mot_data[index][0:self.max_obj]
        new_list = []
        for i in range(len(mot_data_points)):
            new_list.append(mot_data_points[i][0:self.mot_pose_len])
        mot_data_points = new_list
        num_obj = len(mot_data_points)
        seq_len = 50 #len(mot_data_points[0])
        pad_obj_len = max(0,self.max_obj - num_obj)
        pad_seq_len = max(0,self.mot_pose_len - seq_len)
        mot_data_points = np.array(mot_data_points)
        #mot_data_points = np.pad(mot_data_points, ((0,0),(0,pad_seq_len),(0,0)), "edge")
        mot_data_points = np.pad(mot_data_points, ((0,pad_obj_len), (0,0), (0,0)), "constant")
        mot_tensor = torch.from_numpy(mot_data_points).to(torch.float32)

        return rgb_img, lidar_img, goal_tensor, mot_tensor, num_obj

class NavSetDataModule(pl.LightningDataModule):

    def __init__(self,
                 save_data_path: str,
                 train_rosbag_path: str,
                 val_rosbag_path: str,
                 test_rosbag_path: str,
                 batch_size=16,
                 num_workers=8,
                 pin_memory=False,
                 use_weighted_sampling=False,
                 verbose=False):
        
        super().__init__()
        self.save_data_path = save_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_weighted_sampling = use_weighted_sampling
        self.verbose = verbose

        self.train_bags = [b for b in os.listdir(train_rosbag_path) if os.path.isfile(os.path.join(train_rosbag_path,b))]
        print(len(self.train_bags))
        self.val_bags = [b for b in os.listdir(val_rosbag_path) if os.path.isfile(os.path.join(val_rosbag_path,b))]
        print(len(self.val_bags))
        self.test_bags = [b for b in os.listdir(test_rosbag_path) if os.path.isfile(os.path.join(test_rosbag_path,b))]

    def _concatenate_dataset(self, bag_list):
        tmp_sets = []
        for b in bag_list:
            tmp = NavSet(self.save_data_path, b)
            tmp_sets.append(tmp)
        return ConcatDataset(tmp_sets)
    
    def setup(self, stage):

        if stage in (None, "fit"):
            self.train_set = self._concatenate_dataset(self.train_bags)
            self.val_set = self._concatenate_dataset(self.val_bags)
        
        if stage == "validate":
            self.val_set = self._concatenate_dataset(self.val_bags)

        if stage == "test":
            self.test_set = self._concatenate_dataset(self.test_bags)
    
    def train_dataloader(self) -> DataLoader:
        """ return the training dataloader
        """
        return DataLoader(dataset=self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """ return validation dataloader
        """
        return DataLoader(dataset=self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=True) 

    def test_dataloader(self) -> DataLoader:
        """ return validation dataloader
        """
        return DataLoader(dataset=self.test_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=True)    

