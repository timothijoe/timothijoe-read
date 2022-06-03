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

meta_x = 0.8 # m
meta_y  = 0.4 # m
meta_theta = 4.0# degree
meta_vel = 0.5 # m/s
meta_len = 0.5 # m

scale_x = 1 / meta_x 
scale_y = 1 / meta_y 
scale_theta = 1 / meta_theta 
scale_vel = 1 / meta_vel
scale_len = 1 / meta_len

total_traj_num = 0
valid_traj_num = 0

# diction keys order: end_x, end_y, end_theta, start_vel, end_v, total_len

#TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/test_origin_folder/'
#TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/standard_traj_matfiles/'
#TRAJ_LIBRARY_PATH = pwd + '/dataset/eval'
TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/drive_project/ckpt/may10/ckpt'
# TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/variate_len_vae/dataset/eval10'
# if not os.path.exists(TRAJ_LIBRARY_PATH_STANDARD):
#     os.makedirs(TRAJ_LIBRARY_PATH_STANDARD)
TARGET_LIBRARY_PATH = pwd + '/dataset/downsample_len20/'

for cur_file in os.listdir(TRAJ_LIBRARY_PATH):
    cur_traj = os.path.join(TRAJ_LIBRARY_PATH, cur_file)
    traj_file_list.append(cur_traj)



traj_library = {}
traj2_library = {}
for traj_file in traj_file_list:
    print('Preparing for reading path file: {}'.format(traj_file))
    with open(traj_file, 'rb') as file:
        traj_mat = pickle.load(file)

    for key, value in traj_mat.items():
        relative_traj = value['trajectory']
        # traj.shape : 5 x 11, including x, y, theta, vel, time
        end_x = relative_traj[0, -1]
        end_y = relative_traj[1, -1]
        end_theta = relative_traj[2, -1]
        start_vel = relative_traj[3, 0]
        end_vel = relative_traj[3, -1]

        end_x_int = int(end_x * scale_x)
        end_y_int = int(end_y * scale_y)
        end_theta_int = int(end_theta * scale_theta)
        start_vel_int = int(start_vel * scale_vel)
        end_vel_int = int(end_vel * scale_vel)
        standard_theta = float(end_theta_int) * meta_theta

        end_x_key = str(end_x_int)
        end_y_key = str(end_y_int)
        end_theta_key = str(end_theta_int)
        start_vel_key = str(start_vel_int)
        end_vel_key = str(end_vel_int)

        if end_x_key not in traj_library:
            traj_library[end_x_key] = {}
            traj_library[end_x_key][end_y_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = relative_traj 
            total_traj_num += 1
            valid_traj_num +=1
        elif end_y_key not in traj_library[end_x_key]:
            traj_library[end_x_key][end_y_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = relative_traj
            total_traj_num += 1
            valid_traj_num +=1
        elif end_theta_key not in  traj_library[end_x_key][end_y_key]:          
            traj_library[end_x_key][end_y_key][end_theta_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = relative_traj
            total_traj_num += 1
            valid_traj_num +=1
        elif start_vel_key not in traj_library[end_x_key][end_y_key][end_theta_key]:
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = relative_traj
            total_traj_num += 1
            valid_traj_num +=1
        elif end_vel_key not in traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key]:
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = relative_traj
            total_traj_num += 1
            valid_traj_num +=1
        else:
            lib_traj = traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key]
            theta_error_now = np.abs(end_theta - standard_theta)
            theta_error_lib = np.abs(lib_traj[2, -1] - standard_theta)
            if theta_error_now < theta_error_lib:
                traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = relative_traj
            total_traj_num += 1
            continue 
for key_x, value_x in traj_library.items():
    for key_y, value_y in value_x.items():
        for key_theta, value_theta in value_y.items():
            for key_sv, value_sv in value_theta.items():
                for key_ev, value_ev in value_sv.items():
                    library_key = str(len(traj2_library))
                    label_dict = {}
                    label_dict['x'] = key_x 
                    label_dict['y'] = key_y
                    label_dict['theta'] = key_theta 
                    label_dict['sv'] = key_sv 
                    label_dict['ev'] = key_ev 
                    traj2_library[library_key] = {}
                    traj2_library[library_key]['label'] = label_dict 
                    traj2_library[library_key]['trajectory'] = value_ev

pkl_name = 'len_20_new_test.pickle'
pkl_name = TARGET_LIBRARY_PATH + pkl_name
with open(pkl_name, "wb") as fp:
    pickle.dump(traj2_library, fp)
