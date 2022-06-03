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
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep+".")
father_path = pwd
TRAJ_LIBRARY_PATH = father_path + '/data/traj_matfiles'   #remove the last '/'
TRAJ_LIBRARY_PATH_STANDARD = father_path + '/data/standard_traj_matfiles/'   #remove the last '/'
traj_file_list = []


traj_file = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/data/test_origin_folder/test_traj.mat'


def get_random_traj_from_file(traj_file):
    traj_mat = io.loadmat(traj_file)
    traj_mat.pop('__header__')
    traj_mat.pop('__version__')
    traj_mat.pop('__globals__')
    traj_library = {}
    traj_len = len(traj_mat.items())
    print(traj_len)
    traj = None
    zt = 0
    index = np.random.randint(0, traj_len)
    for key, value in traj_mat.items():
        zt += 1
        if zt < index:
            continue
        traj = value
        break 
    print('traj is')
    print(traj.shape)
    print(traj[:, 0])
    print(traj[:, 1])
    print(traj[:, 2])
    print(traj[:, 3])
    print(traj[:, 4])
    return traj
traj = get_random_traj_from_file(traj_file)
relative_trajs = []
absolute_trajs = []
absolute_trajs_ori = []
#for start_index in [0, 4, 8, 12, 16, 20, 24, 28]:
for start_index in [0, 5, 10, 15, 20, 25]:
    end_index = start_index + 11
    # 10 x 2, 0.1s and 
    absolute_value_pos = traj[start_index: end_index, :2]
    absolute_trajs.append(absolute_value_pos)
    init_pos = traj[start_index, :2]
    init_yaw = traj[start_index, 2]
    init_theta = traj[start_index, 2] / 180 * np.pi
    dx = absolute_value_pos[:, 0] - init_pos[0]
    dy = absolute_value_pos[:, 1] - init_pos[1]
    absolute_ori = np.vstack((dx, dy))
    absolute_trajs_ori.append(absolute_ori)
    
    dx_rel = dy * np.sin(init_theta) + dx * np.cos(init_theta)
    dy_rel = dy * np.cos(init_theta) - dx * np.sin(init_theta)
    #relative_pos = np.vstack((dx_rel, dy_rel))
    relative_theta = traj[start_index: end_index, 2] - traj[start_index, 2] 
    relative_velocity = traj[start_index: end_index, 3]
    relative_time = traj[start_index: end_index, 4] - traj[start_index, 4]
    relative_traj = np.vstack((dx_rel, dy_rel, relative_theta, relative_velocity, relative_time))
    relative_trajs.append(relative_traj)





import matplotlib.pyplot as plt
mode = 'draw'
if mode == 'draw':
    fig = plt.figure(figsize = (19, 11))
    fig11 = fig.add_subplot(2, 2, 1)
    fig11.set_xlim([-1, 25])
    fig11.set_ylim([-8, 8])
    fig12 = fig.add_subplot(2, 2, 2)
    fig12.set_xlim([-1, 25])
    fig12.set_ylim([-8, 8])
    fig21 = fig.add_subplot(2, 2, 3)
    fig21.set_xlim([-1, 25])
    fig21.set_ylim([-8, 8])
    fig22 = fig.add_subplot(2, 2, 4)
    fig22.set_xlim([-1, 25])
    fig22.set_ylim([-8, 8])
    norm = plt.Normalize(traj[:,0].min(), traj[:,0].max())
    norm_lon = norm(traj[:,0])
    fig11.scatter(traj[:,0], traj[:,1], c=norm_lon, cmap='viridis')
    for absolute_traj in absolute_trajs:
        fig11.plot(absolute_traj[:, 0], absolute_traj[:, 1])

    zt = 0
    for absolute_traj in absolute_trajs:
        fig12.plot(absolute_traj[:, 0], absolute_traj[:, 1] + zt)
        zt += 0.05

    for relative_traj in relative_trajs:
        fig21.plot(relative_traj[ 0], relative_traj[1])
    
    zt = 0
    for relative_traj in relative_trajs:
        fig22.plot(relative_traj[0], relative_traj[1] + zt)
        zt += 0.1
        print('traj relative yaw:')
        print(relative_traj[2])
        print('traj relative vel:')
        print(relative_traj[3])
        print('traj relative time')
        print(relative_traj[4])
    print(relative_traj.shape)
    image_name = '/home/SENSETIME/zhoutong/hoffnung/vae_trajectory/result/demo/component3.jpg'
    plt.savefig(image_name)
    plt.show()





