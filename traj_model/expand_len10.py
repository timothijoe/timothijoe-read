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

meta_x = 0.2 # m
meta_y  = 0.1 # m
meta_theta = 2.0# degree
meta_vel = 0.1 # m/s
meta_len = 0.2 # m

scale_x = 1 / meta_x 
scale_y = 1 / meta_y 
scale_theta = 1 / meta_theta 
scale_vel = 1 / meta_vel
scale_len = 1 / meta_len

# diction keys order: end_x, end_y, end_theta, start_vel, end_v, total_len
TRAJ_LIBRARY_PATH = pwd + '/dataset/traj_lib/len_10'
TARGET_LIBRARY_PATH = pwd + '/dataset/pruned_traj/'

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
        # if int(label_theta) < -2 or int(label_theta) > 2:
        #     continue
        library_key = str(len(traj2_library))
        traj2_library[library_key] = {}
        traj2_library[library_key]['label'] = label 
        traj2_library[library_key]['traj_type'] = 0 # 0 for 10 and 1 for 20 
        traj_mask = np.ones((5,20))
        traj_mask[:, 9] = 3 # we assign the final point a higher weight
        traj_mask[:, 10:] = 0
        new_traj = np.zeros([5, 21])
        new_traj[:, :11] = value['trajectory']
        fill_element = value['trajectory'][:, -1:]
        fill_block = np.repeat(fill_element, 10, axis=1)
        new_traj[:, 11:] = fill_block
        traj2_library[library_key]['trajectory'] = new_traj
        traj2_library[library_key]['traj_mask'] = traj_mask

pkl_name = 'vae_expand_10.pickle'
pkl_name = TARGET_LIBRARY_PATH + pkl_name
with open(pkl_name, "wb") as fp:
    pickle.dump(traj2_library, fp)
