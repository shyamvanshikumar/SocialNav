import os
import pickle
import glob
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
                 pose_len=30,
                 is_train=True,
                 val_ratio=0.2) -> None:
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

        self.is_train = is_train
        self.pose_data_points = pickle.load(open(self.pose_path, 'rb'))

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

        self.val_start = int(len(os.listdir(self.lidar_dir))*(1-val_ratio))
        if is_train:
            self.length = self.val_start
        else:
            self.length = len(os.listdir(self.lidar_dir)) - self.val_start

    def __len__(self) -> int:
        """ return the length of the of dataset
        """
        return self.length

    def __getitem__(self, index):
        if self.is_train == False:
            index += self.val_start
        rgb_img = Image.open(os.path.join(self.img_dir, f'{index}.png'))
        rgb_img = self.rgb_transforms(rgb_img)

        lidar_img = Image.open(os.path.join(self.lidar_dir, f'{index}.png'))
        lidar_img = self.lidar_transforms(lidar_img)

        curr_pose = self.pose_data_points['pose_sync'][index]
        curr_pose_mat_inv = np.linalg.pinv(get_affine_matrix_quat(curr_pose[0], curr_pose[1], curr_pose[2]))

        goal_points = np.ones((3,len(self.pose_data_points['pose_future'][index])), dtype=np.float32)
        for i, goal_pose in enumerate(self.pose_data_points['pose_future'][index]):
            goal_points[0,i] = goal_pose[0]
            goal_points[1,i] = goal_pose[1]
        
        goal_points =  np.transpose(np.matmul(curr_pose_mat_inv, goal_points)[:-1,:])
        goal_tensor = torch.from_numpy(goal_points).to(torch.float32)

        return rgb_img, lidar_img, goal_tensor



