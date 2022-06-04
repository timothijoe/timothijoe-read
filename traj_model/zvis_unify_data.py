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
meta_y  = 0.3 # m
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
TRAJ_LIBRARY_PATH = pwd + '/dataset/traj_lib/len20'
# TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/drive_project/ckpt/may10/ckpt'
# TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/variate_len_vae/dataset/eval10'
# if not os.path.exists(TRAJ_LIBRARY_PATH_STANDARD):
#     os.makedirs(TRAJ_LIBRARY_PATH_STANDARD)

for cur_file in os.listdir(TRAJ_LIBRARY_PATH):
    cur_traj = os.path.join(TRAJ_LIBRARY_PATH, cur_file)
    traj_file_list.append(cur_traj)
total_num = 0
for traj_file in traj_file_list:
    print('Preparing for reading path file: {}'.format(traj_file))

    with open(traj_file, 'rb') as file:
        traj_mat = pickle.load(file)
    # traj_mat = io.loadmat(traj_file)
    # traj_mat.pop('__header__')
    # traj_mat.pop('__version__')
    # traj_mat.pop('__globals__')
    traj_library = {}
    TRAJ_LIBRARY_PATH_STANDARD = '/home/SENSETIME/zhoutong/hoffnung/Trajectory_VAE/result/image/'
    TRAJ_LIBRARY_PATH_STANDARD_X = TRAJ_LIBRARY_PATH_STANDARD + 'image_x/'
    TRAJ_LIBRARY_PATH_STANDARD_Y = TRAJ_LIBRARY_PATH_STANDARD + 'image_y/'
    TRAJ_LIBRARY_PATH_STANDARD_THETA = TRAJ_LIBRARY_PATH_STANDARD + 'image_theta/'
    label_x_list = []
    label_y_list = []
    label_theta_list = []
    label_sv_list = []
    label_ev_list = []
    for key, value in traj_mat.items():
        total_num += 1
        label = value['label']
        label_x = label['x']  
        label_y = label['y']  
        label_theta = label['theta']  
        label_sv = label['sv']   
        label_ev = label['ev']   
        if label_x not in label_x_list:
            label_x_list.append(label_x)     
        if label_y not in label_y_list:
            label_y_list.append(label_y)   
        if label_theta not in label_theta_list:
            label_theta_list.append(label_theta)     
        if label_sv not in label_sv_list:
            label_sv_list.append(label_sv)  
        if label_ev not in label_ev_list:
            label_ev_list.append(label_ev)  
    print('total trajs: {}'.format(total_num))

    zt = 0   
    for label_xx in label_x_list:
        plt.figure(zt)
        plt.xlim((-1, 22))
        plt.ylim((-9, 9))
        zt += 1
        image_name = label_xx + 'picture.jpg'
        image_name = TRAJ_LIBRARY_PATH_STANDARD_X + image_name
        for key, value in traj_mat.items():
            label = value['label']
            label_x = label['x']
            label_theta = label['theta']
            if label_x == label_xx:
                # if int(label_theta) < -2 or int(label_theta) > 2:
                #     continue
                traj = value['trajectory']
                plt.plot(traj[0], traj[1])
        label = ['x', 'y']
        plt.legend(label, loc='upper left')
        plt.savefig(image_name)  
 
    # for label_yy in label_y_list:
    #     plt.figure(zt)
    #     plt.xlim((-1, 11))
    #     plt.ylim((-4, 4))
    #     zt += 1
    #     image_name = label_yy + 'picture.jpg'
    #     image_name = TRAJ_LIBRARY_PATH_STANDARD_Y + image_name
    #     for key, value in traj_mat.items():
    #         label = value['label']
    #         label_y = label['y']
    #         if label_y== label_yy:
    #             traj = value['trajectory']
    #             plt.plot(traj[0], traj[1])
    #     label = ['x', 'y']
    #     plt.legend(label, loc='upper left')
    #     plt.savefig(image_name)  

 
    # for label_ttheta in label_theta_list:
    #     plt.figure(zt)
    #     plt.xlim((-1, 11))
    #     plt.ylim((-4, 4))
    #     zt += 1
    #     image_name = label_ttheta + 'picture.jpg'
    #     image_name = TRAJ_LIBRARY_PATH_STANDARD_THETA + image_name
    #     for key, value in traj_mat.items():
    #         label = value['label']
    #         label_theta = label['theta']
    #         if label_theta == label_ttheta:
    #             traj = value['trajectory']
    #             plt.plot(traj[0], traj[1])
    #     label = ['x', 'y']
    #     plt.legend(label, loc='upper left')
    #     plt.savefig(image_name)  
    
    
