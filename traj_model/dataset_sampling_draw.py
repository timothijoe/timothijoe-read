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

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



pwd = os.getcwd()
traj_file_list = []

meta_x = 0.2 # m
meta_y  = 0.2 # m
meta_theta = 2.0 # degree
meta_vel = 0.1 # m/s

scale_x = 1 / meta_x 
scale_y = 1 / meta_y 
scale_theta = 1 / meta_theta 
scale_vel = 1 / meta_vel

# diction keys order: end_x, end_y, end_theta, start_vel, end_v, total_len

TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/iros_motion_primitive/data/unify_data/'
#TRAJ_LIBRARY_PATH = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/unify_data/'
for cur_file in os.listdir(TRAJ_LIBRARY_PATH):
    cur_traj = os.path.join(TRAJ_LIBRARY_PATH, cur_file)
    traj_file_list.append(cur_traj)
total_num = 0

traj_mat = {}
max_same_num = 0
for traj_file in traj_file_list:
    print('Preparing for reading path file: {}'.format(traj_file))
    label_x_list = []
    label_y_list = []
    label_theta_list = []
    label_sv_list = []
    with open(traj_file, 'rb') as file:
        traj_mat = pickle.load(file)
    traj_library = {}
    traj_total_num = 0
    for key, value in traj_mat.items():
        total_num += 1
        relative_traj = value['trajectory']

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

        label_x = str(end_x_int)
        label_y = str(end_y_int)
        label_theta = str(end_theta_int)
        label_sv = str(start_vel_int)
        # end_vel_key = str(end_vel_int)
        if label_x not in label_x_list:
            label_x_list.append(label_x)     
        if label_y not in label_y_list:
            label_y_list.append(label_y)   
        if label_theta not in label_theta_list:
            label_theta_list.append(label_theta)     
        if label_sv not in label_sv_list:
            label_sv_list.append(label_sv)  
        
        label_sv = 'start_vel'
 
        if label_sv not in traj_library:
            traj_library[label_sv] = {}
        if label_x not in traj_library[label_sv]:
            traj_library[label_sv][label_x] = {}
        if label_y not in traj_library[label_sv][label_x]:
            traj_library[label_sv][label_x][label_y] = {}
        if label_theta not in traj_library[label_sv][label_x][label_y]:
            traj_library[label_sv][label_x][label_y][label_theta] = {}
            traj_library[label_sv][label_x][label_y][label_theta]['num'] = 0
        traj_library[label_sv][label_x][label_y][label_theta]['num'] += 1
        max_same_num = max(max_same_num,  traj_library[label_sv][label_x][label_y][label_theta]['num'])
        traj_library[label_sv][label_x][label_y][label_theta]['x'] = end_x_int * meta_x
        traj_library[label_sv][label_x][label_y][label_theta]['y'] = end_y_int * meta_y
        traj_library[label_sv][label_x][label_y][label_theta]['theta'] = end_theta_int * meta_theta
        traj_total_num += 1
    #print('len v: {}'.format(len(label_x)))

    fig = plt.figure()
    ax11 = fig.add_subplot(111, projection='3d')
    # ax11 = fig.add_subplot(331, projection='3d')
    # ax12 = fig.add_subplot(332, projection='3d')
    # ax13 = fig.add_subplot(333, projection='3d')
    # ax21 = fig.add_subplot(334, projection='3d')
    # ax22 = fig.add_subplot(335, projection='3d')
    # ax23 = fig.add_subplot(336, projection='3d')
    # ax31 = fig.add_subplot(337, projection='3d')
    # ax32 = fig.add_subplot(338, projection='3d')
    # ax33 = fig.add_subplot(339, projection='3d')


    zt = 0
    max_num = 50
    min_num = 0
    x_l = []
    y_l = []
    theta_l = []
    density_l = []
    for key_sv, value_sv in traj_library.items():
        for key_x ,value_x in value_sv.items():
            for key_y, value_y in value_x.items():
                for key_theta, value_theta in value_y.items():
                    zt +=1
                    point_num = traj_library[key_sv][key_x][key_y][key_theta]['num']
                    norm = point_num / max_num
                    norm = min(1, norm)
                    norm = (1-norm) * 0.7 + 0.3
                    x = traj_library[key_sv][key_x][key_y][key_theta]['x']
                    y = traj_library[key_sv][key_x][key_y][key_theta]['y']
                    theta = traj_library[key_sv][key_x][key_y][key_theta]['theta']
                    x_l.append(x)
                    y_l.append(y)
                    theta_l.append(theta)
                    density_l.append(norm)
    ax11.scatter3D(x_l, y_l, theta_l, s= 1.5, c=density_l, cmap='viridis')
    print('point number: {}'.format(zt))
    plt.show()