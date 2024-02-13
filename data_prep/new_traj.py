# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:31:55 2023

@author: ritik
"""

from sklearn.cluster import DBSCAN
from scipy import ndimage
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os
import yaml
from tqdm import tqdm
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
import pickle

config = yaml.safe_load(open('./config.yaml', 'r'))
res = config['RESOLUTION']
dx = config['LIDAR_BACK_RANGE']/res
dy = config['LIDAR_SIDE_RANGE']/res

def get_affine_matrix_quat(x, y, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])

def map_object_coordinates(prev_, current_, max_dist=5):
    """
    Maps the coordinates of objects from a previous frame to the corresponding objects in the current frame
    based on their Euclidean distance. 

    Args:
    - prev_: list of tuples representing the coordinates of objects in the previous frame
    - current_: list of tuples representing the coordinates of objects in the current frame
    - max_dist: maximum Euclidean distance allowed between two objects in different frames to consider them
    as the same object (default: 10)

    Returns:
    - A dictionary containing the mapped coordinates of objects in the current frame as keys and the 
    corresponding coordinates of objects in the previous frame as values.
    """
    coor_map = dict()
    
    for xy_current in current_:
        dist, c_ = np.inf, [0]
        for xy_prev in prev_:
            d_temp = distance.euclidean(xy_prev, xy_current)
            if(d_temp<max_dist and d_temp<dist):
                dist = d_temp
                c_ = xy_prev
        if(c_ != [0]):
            prev_.remove(c_)
            coor_map[tuple(xy_current)]=tuple(c_)
    
    return coor_map




def obj_detection_and_tracking(sequence, n_frames, buffer = 5):
    """ 
    Object Detection and Tracking Algorithm
    
    This algorithm is used to detect objects in images, track their movement across frames, and store the centroid positions of the objects in a list.
    
    Args:
        sequence (str): Path to the folder containing the image sequence
        buffer (int): The number of frames after which an inactive object should be removed from the list
        
    Outputs:
        inactive_objects (list): Centroids of currently inactive objects detected during previous frames
        active_objects (list): Centroids of currently active objects detected within past 3 frames
    """
    
    # sequence = 'boatbuilding2ahg_spot_data'
    # sequence = 'sac_indoor_jackal_data'
    #sequence = 'jackal_bev_images/ahg2library_data'
    
    
    inactive_objects = [] #centroids of currently inactive objects detected during previous frames
    active_objects = [] #centroids of currently active objects detected within past 3 frames
    
    count = 0
    
    # Loop over each image in the sequence and process it
    for n in range(2, n_frames):
        
        f_name = str(n) + '.png'
        I = Image.open(f'{sequence}/{f_name}').convert('L')
        frame = np.array(I)
        x,y = np.where(frame>0)
        
        points = np.array([[x_, y_] for x_,y_ in zip(x,y) ])
        
        db =  DBSCAN(eps=3, min_samples=20).fit(points)
        labels = db.labels_
        
        no_clusters = len(np.unique(labels) )
        no_noise = np.sum(np.array(labels) == -1, axis=0)
        
        centroids = []
          
        # Create a labeled mask for the image
        mask = np.zeros((frame.shape[0], frame.shape[1]))
        for k in range(len(labels)):
            if(labels[k]!=-1):
                mask[points[k][0]][points[k][1]] = labels[k]+1
    
        labeled_image, num_objects = ndimage.label(mask)
        objs = ndimage.find_objects(labeled_image)
        
        # Extract centroids of each object
        for label in np.unique(mask):
            if label == 0:
                continue # Skip background label
            label_mask = (mask == label)
            size = ndimage.measurements.sum(label_mask)
            if size < 100: # condition set to only detect humans and not larger objects
                centroids.append(ndimage.measurements.center_of_mass(label_mask))
        
        # Display the original frame and the segmented image
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(frame, cmap = plt.cm.gray)
        # axs[0].set_title('2D Lidar BEV')
        # axs[1].imshow(mask)
        # axs[1].set_title('Segmentation mask')
        
        
        count +=1
        
        # If it's the first frame, add all detected objects to the list of active objects
        if(count==1):
            for (x,y) in centroids:
                active_objects.append([(x,y)])
                
            prev_centroids= centroids
        
        else:
            # Map the centroids from the previous frame to the current frame
            coor_map = map_object_coordinates(centroids, prev_centroids, max_dist=15)        
                    
            # Update the list of active objects with the new centroid positions
            for x in active_objects:
                # print(active_objects[i][-1])
                if(x[-1] in coor_map.keys()):
                    x.append(coor_map[x[-1]])
                elif(type(x[-1])!=int):
                    x.append(1)
                elif x[-1] >= buffer:
                    inactive_objects.append(x[:-1])
                    active_objects.remove(x)                    
                elif(x[-2] in coor_map.keys()):
                    new = coor_map[x[-2]]
                    mid = ((x[-2][0]+new[0])/2, (x[-2][1]+new[1])/2)
                    b = x.pop(-1)
                    for n1 in range(b):
                        x.append(mid)
                    x.append(new)
                else:
                    x[-1]+=1
                    
            # Iterate over each centroid in the list
            for c in centroids:
                # Check if the centroid is already in the coordinate map
                if c not in coor_map.values():
                    # If it is not present, add a new object to the active_objects list
                    active_objects.append([0 for n2 in range(n-2)])
                    active_objects[-1].append(c)
                    
            # Create an empty list to store the previous centroids of each active object
            prev_centroids= []
            for x in active_objects:
                if(type(x[-1])!=int):
                    prev_centroids.append(x[-1])
                else:
                    prev_centroids.append(x[-2])
                
        for x in active_objects:        
            if(type(x[-1])==int):
                if (n-1!= len(x) -1 + x[-1]):
                    x[-1]+=1
    
            else:
                if (n-1!= len(x)):
                    try:
                        x.append(coor_map[x[-1]])
                    except KeyError:
                        x.append(x[-1])
    
    # Safety check to catch missed objects
    while(len(active_objects)!=0):
        for x in active_objects:
            if len(x) != count:
                for n3 in range(len(x) - count):
                    x.append(0) 
                
            inactive_objects.append(x)
            active_objects.remove(x)
        
    # Append zeros to make length of each trajectory equal
    for y in inactive_objects:
        if (len(y) != count):
            for n4 in range(count - len(y)):
                y.append(0)
                
    return inactive_objects
    




def plot_tracking_trajectories(sequence, ddata):
    """ Plot trajectories of tracked objects by connecting the coordinate positions
    of each detected object upto the past 'memory' frames"""
    
    # Generate rainbow colors for each object
    color = plt.cm.rainbow(np.linspace(0, 1, len(data)))
    np.random.shuffle(color)
    
    # Get the number of trajectories and the number of steps in each trajectory
    n_traj = len(data)
    memory = 5 # number of past frames to include in the visualization of each trajectory
    
    for j in range(memory, len(data[0])):
        f_name = str(j) + '.png'
        I = Image.open(f'{sequence}/{f_name}')
        I = I.rotate(-90)
        I = ImageOps.mirror(I)
        frame = np.array(I)
        fig, axs = plt.subplots(1, 2, dpi=120) #figsize=(7, 3.5), 
        axs[0].axis('off')
        axs[1].axis('off')
        axs[0].imshow(frame, cmap = plt.cm.gray, origin='lower')
        axs[0].set_title('2D Lidar BEV')
        axs[1].imshow(frame, cmap = plt.cm.gray, origin='lower')
        axs[1].set_title('Tracking Trajectory')
        for i in range(len(data)):
            c = color[i]
            if (data[i][j] != 0):
                l = data[i][j-memory:j+1]
                while 0 in l:
                    l.remove(0)
                
                x, y = [],[]
                if(type(l)!=int):
                    for item in l:
                        x.append(item[0])
                        y.append(item[1])
                
                axs[1].plot(x,y, c=c)
            
            
def plot_all_trajectories(sequence, data, w=401, h=401):           
    """ Compute trajectory of robot from the pose information in the pickle file
        Compute trajectories of humans with respect robot position and plot all trajectories """
    
    
    
    with open(f'D:/CS_ROB/Project/{sequence}.pkl', 'rb') as f:
        pkl_data = pickle.load(f)
    
    poses = pkl_data['pose']
    
    robot_trajectory =[]
    human_trajectories = []
    
    x0, y0 = poses[1][0], poses[1][1]
    for [x, y, z] in poses[1:]:
        robot_trajectory.append((x-x0, y-y0))
    # plt.plot(*zip(*robot_trajectory))
    
    for t in data:
        cur_traj = []
        for idx in range(len(t)):
            if(type(t[idx])!=int):
                cur_traj.append((t[idx][0] - h//2 - robot_trajectory[idx][0], t[idx][1] - w//2 - robot_trajectory[idx][1]))
        human_trajectories.append(cur_traj)
    print('Number of human trajectories = ', len(human_trajectories))     
    color = plt.cm.rainbow(np.linspace(0, 1, len(data)))
    np.random.shuffle(color)
    
    for j in range(len(human_trajectories)):
        c = color[j]
        plt.plot(*zip(*human_trajectories[j]), c=c)
    plt.plot(*zip(*robot_trajectory), color='black', markersize=12)
    

# #sequence = 'jackal_bev_images/ahg2library_data'
# sequence = 'jackal_bev_images/library2pond_data'

# data = obj_detection_and_tracking(sequence, len(os.listdir(sequence)))
# plot_tracking_trajectories(data)
# plot_all_trajectories(sequence, data)

def convert_coordinates(trajectories, pose):
    init_pos = pose[0]
    init_pose_mat_inv = np.linalg.pinv(get_affine_matrix_quat(init_pos[0], init_pos[1], init_pos[2]))
    temp = trajectories
    trajectories = np.array(trajectories)

    for i in range(trajectories.shape[1]):
        curr_pose = pose[i]
        curr_pos_mat = get_affine_matrix_quat(curr_pose[0], curr_pose[1], curr_pose[2])

        all_cur_x = trajectories[:,:,0][:,i]
        all_cur_y = trajectories[:,:,1][:,i]

        all_cur_x = (all_cur_y - dx)*res
        all_cur_y = (dy - all_cur_x)*res

        curr_coor = np.stack((all_cur_x, all_cur_y, np.ones(all_cur_x.shape)))
        global_coor = np.matmul(curr_pos_mat, curr_coor)
        local_coor = np.matmul(init_pose_mat_inv, global_coor)

        trajectories[:,:,0][:,i] = local_coor[0]
        trajectories[:,:,1][:,i] = local_coor[1]
    
    return trajectories

def extract_from_objects(objects, pose_data):
    
    num_of_lidar_img = len(objects[0])
    traj_fram = []
    traj_not_comp = []
    traj_len = 35
    for frm_id in tqdm(range(num_of_lidar_img)):
        traj_len = min(traj_len, num_of_lidar_img - frm_id)
        init_pose = pose_data['pose_sync'][frm_id]

        trajectories = []
        for obj in objects:
            t = []
            cnt_unq = 0
            for idx in range(frm_id, frm_id+traj_len):
                if type(obj[idx]) != int:
                    cnt_unq += 1
                    if len(t) == 0:
                        t = [obj[idx]]*(idx-frm_id+1)
                    else:
                        t.append(obj[idx])
            
            if cnt_unq < 2/3 * traj_len:
                continue

            while len(t) < traj_len:
                t.append(t[-1])

            # # Removing static objects
            # dist = []
            # for i in range(len(t)):
            #     dist.append(math.dist(t[0],t[1]))
            # if np.std(dist) < 0.25:
            #     continue

            trajectories.append(t)
        #print(trajectories)
        traj_not_comp.append(trajectories)
        trajectories = convert_coordinates(trajectories, pose_data['pose_sync'][frm_id:frm_id+traj_len])
        traj_fram.append(trajectories)
    return traj_fram, traj_not_comp

def main():
    traj_len = 35
    train_bags = "../data2/train_bags"
    val_bags = "../data2/val_bags"

    save_data_path = "../data2"
    bags = os.listdir(train_bags) + os.listdir(val_bags)


    for rosbag_path in bags:
        lidar_dir_path = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_lidar_bev'))
        pose_path = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_pose.pkl'))
        mot_pkl_path = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_mot.pkl'))
        pose_data = pickle.load(open(pose_path, 'rb'))
        num_of_lidar_img = len(os.listdir(lidar_dir_path))

        # if not os.path.exists(mov_obj_dir_path):
        #     os.makedirs(mov_obj_dir_path)

        objects = obj_detection_and_tracking(lidar_dir_path, num_of_lidar_img)
        trajectories = extract_from_objects(objects, pose_data)
        print(f"{rosbag_path} moving object extraction begin")
        

                        

