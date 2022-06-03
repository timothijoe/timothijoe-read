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

# Set classification hyper parameters, in order to generate keys of the trajectory
meta_x = 0.4 # m
meta_y  = 0.2 # m
meta_theta = 2.0# degree
meta_vel = 0.2 # m/s
#meta_len = 0.2 # m

scale_x = 1 / meta_x 
scale_y = 1 / meta_y 
scale_theta = 1 / meta_theta 
scale_vel = 1 / meta_vel
#scale_len = 1 / meta_len

total_traj_num = 0
valid_traj_num = 0

# Specify the origin folder and target folder
# Read all the files in origin folder, and writing only one file into target folder
pwd = os.getcwd()
#father_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
TRAJ_LIBRARY_PATH = pwd + '/data/jan_traj_matfiles/'
TRAJ_LIBRARY_PATH_STANDARD = pwd + '/data/unify_data/'
if not os.path.exists(TRAJ_LIBRARY_PATH_STANDARD):
    os.makedirs(TRAJ_LIBRARY_PATH_STANDARD)

traj_file_list = []
for cur_file in os.listdir(TRAJ_LIBRARY_PATH):
    cur_traj = os.path.join(TRAJ_LIBRARY_PATH, cur_file)
    traj_file_list.append(cur_traj)


# read all the trajectory folder files, and restore them into one trajectory
# cut the trajectory into pieces and each piece's inital trajectory is [0,0,0,v]
traj_library = {}
for traj_file in traj_file_list:
    print('Preparing for reading path file: {}'.format(traj_file))
    (prezt, file_path_name) = os.path.split(traj_file)
    path_elem_name = file_path_name.strip('.mat')
    traj_mat = io.loadmat(traj_file)
    traj_mat.pop('__header__')
    traj_mat.pop('__version__')
    traj_mat.pop('__globals__')
    for key, value in traj_mat.items():
        traj = value
        #for start_index in [0, 4, 8, 12, 16, 20, 24, 28]:
        # for start_index in [0, 5, 10, 15, 20, 25]:
        #     end_index = start_index + 11
        for start_index in [0]:
            end_index = start_index + 21
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

traj2_library = {}
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


pkl_name = 'may_28_len20_traj.pickle'
pkl_name = TRAJ_LIBRARY_PATH_STANDARD + pkl_name
with open(pkl_name, "wb") as fp:
    pickle.dump(traj2_library, fp)