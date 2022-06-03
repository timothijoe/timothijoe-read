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
''' data '''
# params.track_lst = '/home/SENSETIME/zhoutong/hoffnung/motion_primitive_library/data/traj_matfiles'
pwd = os.getcwd()
# father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep+".")
# father_path = pwd
# TRAJ_LIBRARY_PATH = father_path + '/data/traj_matfiles'   #remove the last '/'
# TRAJ_LIBRARY_PATH_STANDARD = father_path + '/data/standard_traj_matfiles/'   #remove the last '/'
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
TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/standard_traj_matfiles/'
TRAJ_LIBRARY_PATH_STANDARD = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/filter_traj_matfiles/'
if not os.path.exists(TRAJ_LIBRARY_PATH_STANDARD):
    os.makedirs(TRAJ_LIBRARY_PATH_STANDARD)

for cur_file in os.listdir(TRAJ_LIBRARY_PATH):
    cur_traj = os.path.join(TRAJ_LIBRARY_PATH, cur_file)
    traj_file_list.append(cur_traj)

total_traj_num = 0
valid_traj_num = 0
for traj_file in traj_file_list:
    print('Preparing for reading path file: {}'.format(traj_file))
    (prezt, file_path_name) = os.path.split(traj_file)
    path_elem_name = file_path_name.split('.')[0]
    pkl_name = path_elem_name + '.pickle'
    traj_mat = io.loadmat(traj_file)
    traj_mat.pop('__header__')
    traj_mat.pop('__version__')
    traj_mat.pop('__globals__')
    traj_library = {}
    print(pkl_name)
    for key, value in traj_mat.items():
        traj = value
        # traj.shape : 5 x 11, including x, y, theta, vel, time
        end_x = traj[0, -1]
        end_y = traj[1, -1]
        end_theta = traj[2, -1]
        start_vel = traj[3, 0]
        end_vel = traj[3, -1]

        end_x_int = int(end_x * scale_x)
        end_y_int = int(end_y * scale_y)
        end_theta_int = int(end_theta * scale_theta)
        start_vel_int = int(start_vel * scale_vel)
        end_vel_int = int(end_vel * scale_vel)

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
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = traj 
            total_traj_num += 1
            valid_traj_num +=1
        elif end_y_key not in traj_library[end_x_key]:
            traj_library[end_x_key][end_y_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = traj
            total_traj_num += 1
            valid_traj_num +=1
        elif end_theta_key not in  traj_library[end_x_key][end_y_key]:          
            traj_library[end_x_key][end_y_key][end_theta_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = traj
            total_traj_num += 1
            valid_traj_num +=1
        elif start_vel_key not in traj_library[end_x_key][end_y_key][end_theta_key]:
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key] = {}
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = traj
            total_traj_num += 1
            valid_traj_num +=1
        elif end_vel_key not in traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key]:
            traj_library[end_x_key][end_y_key][end_theta_key][start_vel_key][end_vel_key] = traj
            total_traj_num += 1
            valid_traj_num +=1
        else:
            total_traj_num += 1
            continue 
    pkl_name = TRAJ_LIBRARY_PATH_STANDARD + pkl_name
    with open(pkl_name, "wb") as fp:
        pickle.dump(traj_library, fp)
    print('total number: {}'.format(total_traj_num))
    print('valid number: {}'.format(valid_traj_num))