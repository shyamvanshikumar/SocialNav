import os
import cv2
import yaml
import math
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

config = yaml.safe_load(open('../config.yaml', 'r'))
res = config['RESOLUTION']
dx = config['LIDAR_BACK_RANGE']/res
dy = config['LIDAR_SIDE_RANGE']/res

def get_affine_matrix_quat(x, y, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])

def process_img(path):
    """ Takes image path as input and returns list of centroids """

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)
    img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]

    output = cv2.connectedComponentsWithStats(img, 8)
    (numLabels, labels, stats, centroids) = output

    objs = []
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if((area>10) and (area<70)):
            centroids[i][0] = (centroids[i][0] - dx)*res
            centroids[i][1] = (dy - centroids[i][1])*res
            objs.append(centroids[i])
    #print(len(objs))
    return objs



def extract_traj(objects, init_idx, poses, traj_len):
    init_pos = poses[0]
    init_pose_mat_inv = np.linalg.pinv(get_affine_matrix_quat(init_pos[0], init_pos[1], init_pos[2]))
    objs_traj = []

    for ts in range(init_idx, init_idx + traj_len):
        objs = objects[ts]
        if(len(objs) == 0):
            continue

        objs = np.array(objs)
        a = np.ones((objs.shape[0],1))
        objs_loc1 = np.transpose(np.concatenate((objs,a),axis=1))
        curr_pose = poses[ts-init_idx]
        curr_pos_mat = get_affine_matrix_quat(curr_pose[0], curr_pose[1], curr_pose[2])
        objs_glb = np.matmul(curr_pos_mat, objs_loc1)
        objs_init_loc = np.matmul(init_pose_mat_inv, objs_glb)
        objs_curr_pos = np.transpose(objs_init_loc[:-1,:])
        
        prev_obj_pres = [False]*len(objs_traj)
        prev_len = len(objs_traj)
        for obj_cur_pos in objs_curr_pos:
            clos_idx = -1
            closest = 0.5
            for j in range(prev_len):
                cur_dist = math.dist(objs_traj[j][-1], obj_cur_pos)
                if(cur_dist < closest):
                    clos_idx = j

            if clos_idx != -1:
                objs_traj[clos_idx].append(obj_cur_pos)
                prev_obj_pres[clos_idx] = True

            else:
                new_traj = [obj_cur_pos]*(ts - init_idx + 1)
                objs_traj.append(new_traj)
        
        #print("1", len(objs_traj[0]))
        for j in range(prev_len):
            if prev_obj_pres[j] == False:
                objs_traj[j].append(objs_traj[j][-1])
        #print("2", len(objs_traj[0]))
    
    #remove static objects
    mov_obj_traj = []
    for traj in objs_traj:
        dist = []
        for i in range(len(traj)):
            dist.append(math.dist(traj[0], traj[i]))
        if np.std(dist) > 1.00:
            mov_obj_traj.append(traj[0:traj_len])

    return mov_obj_traj


if __name__ == "__main__":
    # pose_data = pickle.load(open("../data/A_Spot_Library_Dobie_Wed_Nov_10_57_pose.pkl", 'rb'))
    # idx =  100
    # print(pose_data['pose_sync'][idx])
    # poses = pose_data['pose_sync'][idx:idx+21]
    # traj = extract_traj("../data", "A_Spot_Library_Dobie_Wed_Nov_10_57.bag", idx, poses)
    #print(traj)

    traj_len = 50
    train_bags = "../data2/train_bags"
    val_bags = "../data2/val_bags"

    save_data_path = "../data2"
    bags = os.listdir(train_bags) + os.listdir(val_bags)
    for rosbag_path in bags:
        print(f"{rosbag_path} moving object extraction begin")
        lidar_dir_path = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_lidar_bev'))
        pose_path = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_pose.pkl'))
        mov_obj_dir_path = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_mot'))
        pose_data = pickle.load(open(pose_path, 'rb'))
        num_of_lidar_img = len(os.listdir(lidar_dir_path))

        if not os.path.exists(mov_obj_dir_path):
            os.makedirs(mov_obj_dir_path)
        
        #process all images
        print("Processing images!!!!")
        objects = []
        for ts in tqdm(range(num_of_lidar_img)):
            img_path = os.path.join(lidar_dir_path, f"{ts}.png")
            objs = process_img(img_path)
            objects.append(objs)

        for idx in tqdm(range(num_of_lidar_img)):
            traj_len = min(50, num_of_lidar_img - idx)
            poses = pose_data['pose_sync'][idx:idx+traj_len]
            curr_traj = extract_traj(objects, idx, poses, traj_len)
            mov_obj_data_path = os.path.join(mov_obj_dir_path, f"{idx}.pkl")
            pickle.dump(curr_traj, open(mov_obj_data_path, 'wb'))
        
        print(f"Extraction complete... Saved to {mov_obj_dir_path}")
        


        