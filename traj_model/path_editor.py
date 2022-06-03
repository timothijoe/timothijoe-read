import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from matplotlib import cm
import scipy.io as io
import copy

class PathParam():
    def __init__(self, lat0, yaw0, lon1, lat1, yaw1, lon_final=80):
        '''
        yaw: degree, need to /180 * math.pi
            cubic polunomial curve: lat = a0 + a1 * lon + a2 * lon^2 + a3 * lon^3
            augument:
                lat0: current lateral position
                yaw0: current yaw angle, in degree
                lon1: ending longitudinal position
                lat1: ending lateral position
                yaw1: ending yaw angle, in degree
                lon_final: the longitudinal distance horizon
            return:
                lon: lateral position (with precision of 0.1m)
                lat: lateral position (corresponding to lon)
        '''
        yaw0 = math.tan(yaw0/180*math.pi)
        yaw1 = math.tan(yaw1/180*math.pi)
        self.yaw1 = yaw1
        self.Horizon = lon1 
        self.lon_final = lon_final 
        self.a0 = lat0
        self.a1 = yaw0
        self.a3 = (2*yaw0 + self.a1*self.Horizon + yaw1*self.Horizon - 2*lat1) / (self.Horizon**3)
        self.a2 = (yaw1 - self.a1 - 3 * self.a3 * (self.Horizon**2) ) / (2*self.Horizon)
        self.illegal = False
        self.extend_require = True
        self.GetPathProfile()

    def GetPathProfile(self):
        # calculate each segment length
        cal_length = lambda pos:(np.sum(np.sqrt(np.sum(np.square((pos[1:,:2] - pos[:-1,:2])),axis=1)),axis=0))
        # longitude precision: 0.1
        self.lon = np.arange(self.Horizon*10) / 10
        # calculate the corresponding latitude according for longitude length by precision
        self.lat = self.a0 + self.a1 * self.lon + self.a2 * (self.lon**2) + self.a3 * (self.lon**3)
        # expand dimension for stack
        self.lon = np.expand_dims(self.lon,1)
        self.lat = np.expand_dims(self.lat,1)
        if self.extend_require:
            for i in range(100):
                self.lon = np.vstack((self.lon, self.lon[-1]+0.1))
                self.lat = np.vstack((self.lat, self.lat[-1] + 0.1 * np.tan(self.yaw1)))
        self.lon = self.lon * 0.6
        
        # The following trajectory cannot guarantee kinematic constraints
        # while self.lon[-1] < self.lon_final:
        #     self.lon = np.vstack((self.lon, self.lon[-1]+0.1))
        #     self.lat = np.vstack((self.lat, self.lat[-1] + 0.1 * np.tan(self.yaw1)))
        self.yaw = np.arctan((self.lat[1:] - self.lat[:-1]) / (self.lon[1:] - self.lon[:-1])) / math.pi * 180
        self.yaw = np.vstack((self.yaw, self.yaw[-1]))
        self.path = np.hstack((self.lon, self.lat, self.yaw))
        self.path_length = [cal_length(self.path[:i+1,:2]) for i in range(len(self.path-1))]
        # Verify it the initial yaw legal or not
        if self.path[0,2] > 30 or self.path[0,2] < -30:
            self.illegal = True
            return 
        for i in range(1, self.path.shape[0]):
            yaw_diff = self.path[i,2] - self.path[i-1,2] 
            if yaw_diff < -2 or yaw_diff > 2:
                self.illegal = True 
                break

    def GetPosFromLength(self, s):
        # given velocity profile, we can get the path length, and get the correct position in the path, even more precise than the path precision
        cal_length = lambda pos:(np.sum(np.sqrt(np.sum(np.square((pos[1:,:2] - pos[:-1,:2])),axis=1)),axis=0))
        matched_ind = min(np.where(np.array(self.path_length)-s >= 0)[0])
        if matched_ind == 0:
            return self.path[:1,:]
        else:
            pos = np.zeros((1,3))         
            percent = (s - cal_length(self.path[:matched_ind, :])) / cal_length(self.path[matched_ind-1:matched_ind+1, :])
            pos[0,0] = self.path[matched_ind-1, 0] + percent * (self.path[matched_ind, 0] - self.path[matched_ind-1, 0])
            pos[0,1] = self.path[matched_ind-1, 1] + percent * (self.path[matched_ind, 1] - self.path[matched_ind-1, 1])
            pos[0,2] = self.path[matched_ind-1, 2] + percent * (self.path[matched_ind, 2] - self.path[matched_ind-1, 2])

            return pos
    def PlotPath(self):
        plt.title('Path Profile')
        plt.plot(self.path[:,0], self.path[:,1])

def PlotSinglePath(lat0 = 0, lat1=4, yaw0 = 0, yaw1 = 15, lon1=20):
    PathProfile = PathParam(lat0 = lat0, lat1= lat1, yaw0 = yaw0, yaw1=yaw1, lon1=lon1)
    if PathProfile.illegal == True:
        return
    plt.plot(PathProfile.path[:,0], PathProfile.path[:,1])

# def storePathtoDict(path_dict, path_id, lat0 = 0, lat1=4, yaw0 = 0, yaw1 = 0, lon1=20):
#     PathProfile = PathParam(lat0 = lat0, lat1= lat1, yaw0 = yaw0, yaw1=yaw1, lon1=lon1)
#     if PathProfile.illegal == True:
#         return
#     path_dict[path_id] = PathProfile.path
def getPath(lat0 = 0, lat1=4, yaw0 = 0, yaw1 = 15, lon1=20):
    PathProfile = PathParam(lat0 = lat0, lat1= lat1, yaw0 = yaw0, yaw1=yaw1, lon1=lon1)
    if PathProfile.illegal == True:
        return None
    else:
        return PathProfile

lane_width = 3.5
lane_num = 5
road_width = lane_width * lane_num
road_delta = 0.25
road_lat_lst = [-road_width/2 + i*road_delta for i in range(int(road_width/road_delta) + 1)]
max_speed = 10
time_horizon = 5
lon_dist_horizon = time_horizon * max_speed 
lon_dist_delta =1.0 # 0.5

lon_dist_horizon_lst = [i*lon_dist_delta for i in range(int(lon_dist_horizon/lon_dist_delta) + 1)]


pwd = os.getcwd()
PATH_LIBRARY_PATH = pwd + '/data/jan_path_matfiles/'
print(PATH_LIBRARY_PATH)
if not os.path.exists(PATH_LIBRARY_PATH):
    os.makedirs(PATH_LIBRARY_PATH)
# mode = 'draw' 
mode = 'save'

if mode == 'draw':
    path_id = 0
    #degree_list = [60,50,40,30,20,10,0,-10,-20,-30,-40,-50,-60]
    degree_list = [0, 5, 10]
    for degree in degree_list:
        path_dictionary = {}
        for long_dist in lon_dist_horizon_lst[1:]:
            ind = 0
            for lat_dist in road_lat_lst:
                ind = ind + 1
                # if degree != 0:
                #     continue
                PlotSinglePath(lat1 = lat_dist, lon1=long_dist, yaw1 = degree)
        plt.grid()
        plt.show()

if mode == 'save':
    path_id = 0
    #degree_list = [60,50,40,30,20,10,0,-10,-20,-30,-40,-50,-60]
    #degree_list = [30,15,0,-15,-30]
    degree_list = [10, 5 ,0,-5. -10]
    #degree_list = [0]
    for degree in degree_list:
        print('preparing degree {}'.format(degree))
        path_dictionary = {}
        for long_dist in lon_dist_horizon_lst[1:]:
            for lat_dist in road_lat_lst:
                path = getPath(lat1 = lat_dist, lon1=long_dist, yaw1=degree)
                if path is not None:
                    path_dictionary[str(path_id)] = copy.deepcopy(path.path)
                    path_id += 1 
        mat_name = PATH_LIBRARY_PATH + "degree" + str(degree) + "mat.mat"   
        io.savemat(mat_name, path_dictionary) 
