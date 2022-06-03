from torch.utils.data import Dataset, DataLoader
import torch
import os
import pdb
import glob
import numpy as np
import random
import copy
import scipy.io as io
import pickle
import matplotlib.pyplot as plt
''' data '''
# params.track_lst = '/home/SENSETIME/zhoutong/hoffnung/motion_primitive_library/data/traj_matfiles'
pwd = os.getcwd()
traj_file_list = []

meta_x = 0.6 # m
meta_y  = 0.3 # m
meta_theta = 2.0# degree
meta_vel = 0.2 # m/s
meta_len = 0.2 # m

scale_x = 1 / meta_x 
scale_y = 1 / meta_y 
scale_theta = 1 / meta_theta 
scale_vel = 1 / meta_vel
scale_len = 1 / meta_len

# diction keys order: end_x, end_y, end_theta, start_vel, end_v, total_len

#TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/test_origin_folder/'
#TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/standard_traj_matfiles/'
#TRAJ_LIBRARY_PATH = pwd + '/dataset/eval'
TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/drive_project/ckpt/may10/ckpt'
TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/dataset/downsample_len20'
TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/dataset/may30'
# TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/variate_len_vae/dataset/eval10'
# if not os.path.exists(TRAJ_LIBRARY_PATH_STANDARD):
#     os.makedirs(TRAJ_LIBRARY_PATH_STANDARD)
TARGET_LIBRARY_PATH = pwd + '/dataset/prune_len20/'

for cur_file in os.listdir(TRAJ_LIBRARY_PATH):
    cur_traj = os.path.join(TRAJ_LIBRARY_PATH, cur_file)
    traj_file_list.append(cur_traj)




traj2_library = {}
for traj_file in traj_file_list:
    print('Preparing for reading path file: {}'.format(traj_file))
    with open(traj_file, 'rb') as file:
        traj_mat = pickle.load(file)

    for key, value in traj_mat.items():
        label = value['label']
        label_x = label['x']  
        label_y = label['y']  
        label_theta = label['theta']  
        label_sv = label['sv']   
        label_ev = label['ev']
        traj_mask = np.ones((5,20))
        traj_mask[:, 19] = 3
        # if int(label_theta) < -10 or int(label_theta) > 10:
        #     continue
        # if int(label_y) % 3 != 0:
        #     continue
        library_key = str(len(traj2_library))
        traj2_library[library_key] = {}
        traj2_library[library_key]['label'] = label 
        traj2_library[library_key]['traj_type'] = 1
        traj2_library[library_key]['trajectory'] = value['trajectory']
        traj2_library[library_key]['traj_mask'] = traj_mask

pkl_name = 'may30_vae_prune_20.pickle'
pkl_name = TARGET_LIBRARY_PATH + pkl_name
with open(pkl_name, "wb") as fp:
    pickle.dump(traj2_library, fp)
