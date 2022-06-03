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

#TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/test_origin_folder/'
#TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/standard_traj_matfiles/'
TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/filter_traj_matfiles/'
# if not os.path.exists(TRAJ_LIBRARY_PATH_STANDARD):
#     os.makedirs(TRAJ_LIBRARY_PATH_STANDARD)

for cur_file in os.listdir(TRAJ_LIBRARY_PATH):
    cur_traj = os.path.join(TRAJ_LIBRARY_PATH, cur_file)
    traj_file_list.append(cur_traj)

for traj_file in traj_file_list:
    print('Preparing for reading path file: {}'.format(traj_file))

    with open(traj_file, 'rb') as file:
        traj_mat = pickle.load(file)
    # traj_mat = io.loadmat(traj_file)
    # traj_mat.pop('__header__')
    # traj_mat.pop('__version__')
    # traj_mat.pop('__globals__')
    traj_library = {}
    TRAJ_LIBRARY_PATH_STANDARD = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/result/image/'
    zt = 0
    for key_x, value_x in traj_mat.items():
        plt.figure(zt)
        plt.xlim((-1, 11))
        plt.ylim((-4, 4))
        zt += 1
        image_name = key_x + 'picture.jpg'
        image_name = TRAJ_LIBRARY_PATH_STANDARD + image_name
        #fig = plt.figure(figsize = (19, 11))
        for key_y, value_y in value_x.items():
            for key_theta, value_theta in value_y.items():
                for key_sv, value_sv in value_theta.items():
                    for key_ev, value_ev in value_sv.items():
                        library_key = str(len(traj_library))
                        plt.plot(value_ev[0], value_ev[1])
        label = ['x', 'y']
        plt.legend(label, loc='upper left')
        plt.savefig(image_name)

    
    