#!/usr/bin/env python3

import os
import cv2
import pickle
import math
import yaml
import rospy
import rosbag
import subprocess
import numpy as np
import message_filters
from parser_utils import BEVLidar
from termcolor import cprint
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, CompressedImage
from tqdm.auto import tqdm

class ParseBag:
    def __init__(self, config_path, odom_msgs, odom_time_stamps):
        #self.rosbag_play_process = rosbag_play_process
        self.odom_msgs = odom_msgs
        self.odom_ts = np.asarray(odom_time_stamps)

        lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
        rgb_img = message_filters.Subscriber('/image_raw/compressed', CompressedImage)
        odom = message_filters.Subscriber('/odom', Odometry)

        ts = message_filters.ApproximateTimeSynchronizer(
            [lidar, rgb_img, odom], 100, 0.05, allow_headerless=True)
        ts.registerCallback(self.callback)

        self.config = yaml.safe_load(open(config_path, 'r'))
        print('Config file loaded!')
        print(self.config)

        self.pose_data = {'pose_future': [], 'pose_history': []}
        self.sync_odoms = []
        self.lidar_imgs = []
        self.rgb_imgs = []

        # lidar data handler
        self.bevlidar_handler = BEVLidar(x_range=(-self.config['LIDAR_BACK_RANGE'], self.config['LIDAR_FRONT_RANGE']),
                                         y_range=(-self.config['LIDAR_SIDE_RANGE'], self.config['LIDAR_SIDE_RANGE']),
                                         z_range=(-self.config['LIDAR_BOTTOM_RANGE'], self.config['LIDAR_TOP_RANGE']),
                                         resolution=self.config['RESOLUTION'], threshold_z_range=False)
        
    def callback(self, lidar_sync, rgb_img_sync, odom_sync):
        """ callback function for apprx. time synchronizer """

        # current time is based on the current lidar img
        current_time = lidar_sync.header.stamp.to_sec()

        # find the range of future message (groud truth of prediction) indexes in the recorded odom messages
        odom_closest_index = np.searchsorted(self.odom_ts, current_time)+1
        odom_future_index = min(odom_closest_index + 52, len(self.odom_msgs) - 1)
        odom_past_index = max(odom_closest_index - 10, 0)

        pose_future = self.convert_odom_to_pose(self.odom_msgs[odom_closest_index:odom_future_index])
        pose_history = self.convert_odom_to_pose(self.odom_msgs[odom_past_index:odom_closest_index])
        self.pose_data['pose_future'].append(pose_future)
        self.pose_data['pose_history'].append(pose_history)

        self.sync_odoms.append(odom_sync)
        
        # process lidar pointcloud to get bev
        lidar_points = pc2.read_points(
            lidar_sync, skip_nans=True, field_names=("x", "y", "z"))
        bev_lidar_image = self.bevlidar_handler.get_bev_lidar_img(lidar_points)
        self.lidar_imgs.append(bev_lidar_image)

        # rgb_img
        self.rgb_imgs.append(rgb_img_sync.data)

    def convert_odom_to_pose(self, odoms):
        tmp = []
        for odom in odoms:
            tmp.append([odom.pose.pose.position.x, odom.pose.pose.position.y,
                        [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
                         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]])
        return tmp

    def save_data(self, rosbag_path, save_data_path):

        pose_path = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_pose.pkl'))
        lidar_dir = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_lidar_bev'))
        img_dir = os.path.join(save_data_path, rosbag_path.split('/')[-1].replace('.bag','_rgb_img'))

        self.pose_data['pose_sync'] = self.convert_odom_to_pose(self.sync_odoms)
        print('Saving pose to: ', pose_path)
        pickle.dump(self.pose_data, open(pose_path, 'wb'))

        
        print('Saving lidar images to:', lidar_dir)
        if not os.path.exists(lidar_dir):
            os.makedirs(lidar_dir)
        
        for i, lidar_img in enumerate(self.lidar_imgs):
            file_path = os.path.join(lidar_dir, f'{i}.png')
            if not cv2.imwrite(file_path, lidar_img):
                raise Exception('Could not write image')
            print("saved to ", file_path)

        
        print('Saving rgb images to:', img_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        
        for i, rgb_img in enumerate(self.rgb_imgs):
            file_path = os.path.join(img_dir, f'{i}.png')
            a = np.frombuffer(rgb_img, np.uint8)
            mat = cv2.imdecode(a, cv2.IMREAD_COLOR)
            if not cv2.imwrite(file_path, mat):
                raise Exception('Could not write image')
            print("saved to ", file_path)
        
        cprint('Done!', 'green')

def main():
    rospy.init_node('rosbag_parser', anonymous=True)
    rosbag_path = rospy.get_param('rosbag_path')
    save_data_path = rospy.get_param('save_data_path')
    config_path = rospy.get_param('config_path')

    # check if path to rosbag exists
    if not os.path.exists(rosbag_path):
        raise Exception('invalid rosbag path')
    
    # check if the save_data_path exists
    # create directory if needed
    if not os.path.exists(save_data_path):
        cprint('Creating directory : ' +
               save_data_path, 'blue', attrs=['bold'])
        os.makedirs(save_data_path)
    else:
        cprint('Directory already exists : ' +
               save_data_path, 'blue', attrs=['bold'])
    
    # parse the rosbag file and extract the odometry data
    cprint('First reading all the odom messages and timestamps from the rosbag',
           'green', attrs=['bold'])
    ros_bag = rosbag.Bag(rosbag_path)
    info_dict = yaml.safe_load(ros_bag._get_yaml_info())
    duration = info_dict['end'] - info_dict['start']
    print('rosbag_length: ', duration)

    # read all the odometry messages
    odom_msgs, odom_time_stamps = [], []
    for topic, msg, t in tqdm(ros_bag.read_messages(topics=['/odom'])):
        odom_msgs.append(msg)
        if len(odom_time_stamps) == 0:
            odom_time_stamps.append(0.0)
            start_time = t.to_sec()
        else:
            odom_time_stamps.append(t.to_sec())
    cprint('Done reading odom messages from the rosbag !!!',
           color='green', attrs=['bold'])
    
    # instantiate the parser class object
    dataParser = ParseBag(config_path, odom_msgs, odom_time_stamps)

    # skip last six seconds and first ten seconds
    skip_end_sec = 6
    skip_start_sec = 10
    play_duration = str(
        int(math.floor(duration) - skip_end_sec - skip_start_sec))
    cprint(f'play duration: {play_duration} s', 'green', attrs=['bold'])

    rosbag_play_process = subprocess.Popen([
        'rosbag', 'play', rosbag_path, '-r', '1.0', '--clock', '-u',
        play_duration, '-s',
        str(skip_start_sec)
    ])

    while not rospy.is_shutdown():
        # check if the python process is still running
        if rosbag_play_process.poll() is not None:
            print('rosbag process has stopped')
            dataParser.save_data(rosbag_path, save_data_path)
            print('Data was saved in :: ', save_data_path)
            exit(0)

    rospy.spin()

if __name__ == "__main__":
    main()
