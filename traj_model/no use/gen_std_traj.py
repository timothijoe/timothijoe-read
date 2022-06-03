from torch.utils.data import Dataset, DataLoader
import torch
import os
import pdb
import glob
import numpy as np
import random
import copy
import scipy.io as io

''' data '''
# params.track_lst = '/home/SENSETIME/zhoutong/hoffnung/motion_primitive_library/data/traj_matfiles'
pwd = os.getcwd()
# father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep+".")
# father_path = pwd
# TRAJ_LIBRARY_PATH = father_path + '/data/traj_matfiles'   #remove the last '/'
# TRAJ_LIBRARY_PATH_STANDARD = father_path + '/data/standard_traj_matfiles/'   #remove the last '/'
traj_file_list = []

TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/test_origin_folder/'
TRAJ_LIBRARY_PATH_STANDARD = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/standard_traj_matfiles/'
if not os.path.exists(TRAJ_LIBRARY_PATH_STANDARD):
    os.makedirs(TRAJ_LIBRARY_PATH_STANDARD)

for cur_file in os.listdir(TRAJ_LIBRARY_PATH):
    cur_traj = os.path.join(TRAJ_LIBRARY_PATH, cur_file)
    traj_file_list.append(cur_traj)

for traj_file in traj_file_list:
    print('Preparing for reading path file: {}'.format(traj_file))
    (prezt, file_path_name) = os.path.split(traj_file)
    path_elem_name = file_path_name.strip('.mat')
    traj_mat = io.loadmat(traj_file)
    traj_mat.pop('__header__')
    traj_mat.pop('__version__')
    traj_mat.pop('__globals__')
    traj_library = {}
    for key, value in traj_mat.items():
        traj = value
        #for start_index in [0, 4, 8, 12, 16, 20, 24, 28]:
        for start_index in [0, 5, 10, 15, 20, 25]:
            end_index = start_index + 11
            # 10 x 2, 0.1s and 
            absolute_value_pos = traj[start_index: end_index, :2]
            init_pos = traj[start_index, :2]
            init_theta = traj[start_index, 2] / 180 * np.pi
            dx = absolute_value_pos[:, 0] - init_pos[0]
            dy = absolute_value_pos[:, 1] - init_pos[1]
            
            dx_rel = dy * np.sin(init_theta) + dx * np.cos(init_theta)
            dy_rel = dy * np.cos(init_theta) - dx * np.sin(init_theta)
            #relative_pos = np.vstack((dx_rel, dy_rel))
            relative_theta = traj[start_index: end_index, 2] - traj[start_index, 2] 
            relative_velocity = traj[start_index: end_index, 3]
            relative_time = traj[start_index: end_index, 4] - traj[start_index, 4]
            relative_traj = np.vstack((dx_rel, dy_rel, relative_theta, relative_velocity, relative_time))
            library_key = len(traj_library)
            traj_library[str(library_key)] = relative_traj
    mat_name = TRAJ_LIBRARY_PATH_STANDARD + file_path_name
    io.savemat(mat_name, traj_library)
            