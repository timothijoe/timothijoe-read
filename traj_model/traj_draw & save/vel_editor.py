import math
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from matplotlib import cm
import scipy.io as io

class SpeedParam():
    def __init__(self, v0=0, acc0=0, v1=0, v2=0, stop_time=2, speed_pattern='forward', T=4, dt=0.1, composed=True):
        '''
            cubic polynomial speed profile: v = a0 + a1 * t + a2 * t^2 + a3 * t^3
            augument:
                v0: current speed
                acc0: current acceleration
                v1: at 'forward' pattern, the ending speed
                stop_time: at 'brake' pattern, the brake time
                speed_pattern: pattern of speed profile, 'brake' or 'forward'
                T: time horizon
            return:
                self.t: time steps
                self.s: speed at each time steps
            Note: To guarantee velocity profile stability, T must be times of 2 and dt must be times of 0.005
        '''
        self.T = T
        self.dt = dt
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.T_meta = 1.0 
        self.dt_meta = 0.005
        # assert int(self.T / (2 * self.T_meta) * 10) % 10  == 0
        # assert int(self.dt / self.dt_meta) * 10) % 10== 0
        self.stop_time = stop_time
        self.speed_pattern = speed_pattern
        self.illegal = False
        if composed is True:
            self.T = self.T
        self.test_illegal()


        if self.speed_pattern == 'forward':
            self.a0 = v0
            self.a1 = acc0
            self.a3 = (self.a1 * self.T_meta + 2 * self.a0 - 2 * v1) / (self.T_meta**3)
            self.a2 = (-self.a1 - 3 * self.a3 * (self.T_meta**2)) / (2 * self.T_meta)
            
            self.a4 = v1
            self.a5 = acc0 
            self.a7 = (self.a5 * self.T_meta + 2 * self.a4 - 2 * v2) / (self.T_meta**3)
            self.a6 = (-self.a5 - 3 * self.a7 * (self.T_meta**2)) / (2 * self.T_meta)
        else:
            self.a0 = v0
            self.a1 = acc0
            self.a3 = (2 * self.a0 + self.a1 * stop_time) / (stop_time ** 3)
            self.a2 = (-self.a1 - 3 * self.a3 * stop_time**2) / (2 * stop_time)

        self.GetSpeedProfile()

    def GetSpeedProfile(self):
        th = int(self.T / self.dt)
        if self.speed_pattern == 'forward':
            self.t_meta = np.arange(self.T_meta / self.dt_meta) * self.dt_meta
            self.s_meta = self.a0 + self.a1 * self.t_meta + self.a2 * (self.t_meta**2) + self.a3 * (self.t_meta**3)
            self.s_append = self.a4 + self.a5 * self.t_meta + self.a6 * (self.t_meta**2) + self.a7 * (self.t_meta**3)


        # if self.speed_pattern == 'forward':
        #     self.t = np.arange(self.T / self.dt) * self.dt
        #     self.s = self.a0 + self.a1 * self.t + self.a2 * (self.t**2) + self.a3 * (self.t**3)
        #     self.s_append = self.a4 + self.a5 * self.t + self.a6 * (self.t**2) + self.a7 * (self.t**3)
        else:
            t = np.arange((self.stop_time+0.1)*10) / 10
            s = self.a0 + self.a1 * t + self.a2 * (t**2) + self.a3 * (t**3)            
            self.t = np.arange(int((self.T+0.1)*10)) / 10
            self.s = np.array([s[i] if i < len(s) else 0 for i in range(int((self.T+0.1)*10)) ])
        self.t_append = np.arange(self.T_meta / self.dt_meta) * self.dt_meta + self.T_meta
        self.t_meta= np.hstack((self.t_meta, self.t_append[1:]))
        self.s_meta= np.hstack((self.s_meta, self.s_append[1:])) 
        t_skip = int(self.dt / ( self.dt_meta))
        t_scale = self.T / (self.T_meta * 2)
        t_scale_int = int(t_scale)
        t_skip = int (t_skip / t_scale_int)
        # assert s_skip % t_skip ==0:
        # self.t = self.t_meta[::t_skip] * t_scale
        #self.t = np.arange(self.T / self.dt) * self.dt

        self.s = self.s_meta[::t_skip]
        t_len = int(self.T / self.dt)
        s_len = len(self.s)
        vel_len = min(t_len, s_len)
        self.t = np.arange(vel_len) * self.dt
        self.s = self.s[:vel_len]
        self.vel_list = np.vstack((self.t,self.s))
        #print(self.vel_list.shape)
        return self.t, self.s

    def test_illegal(self):
        if abs(self.v0 - self.v1) > 5 * self.T /2:
            self.illegal = True 
        elif abs(self.v1 - self.v2) > 5 * self.T / 2:
            self.illegal = True 

    def GetSpeed(self, t):
        if t > self.T:
            raise Exception ('GetSpeed method: The specified time exceeds the time horizon of speed profile')

        if self.speed_pattern == 'forward':
            return self.a0 + self.a1 * t + self.a2 * (t**2) + self.a3 * (t**3)
        else:
            if t <= self.stop_time:
                return self.a0 + self.a1 * t + self.a2 * (t**2) + self.a3 * (t**3)
            else:
                return self.a0 + self.a1 * self.stop_time + self.a2 * (self.stop_time**2) + self.a3 * (self.stop_time**3)

    def GetDistance(self, t):
        if t > self.T:
            raise Exception ('GetDistance method: The specified time exceeds the time horizon of speed profile')

        if self.speed_pattern == 'forward':
            return 1.0 / 4.0 * self.a3 * pow(t,4) + 1.0 / 3.0 * self.a2 * pow(t,3) + 1.0 / 2.0 * self.a1 * pow(t,2) + self.a0 * t
        else:
            if t <= self.stop_time:
                return 1.0 / 4.0 * self.a3 * pow(t,4) + 1.0 / 3.0 * self.a2 * pow(t,3) + 1.0 / 2.0 * self.a1 * pow(t,2) + self.a0 * t
            else:
                return 1.0 / 4.0 * self.a3 * pow(self.stop_time,4) + 1.0 / 3.0 * self.a2 * pow(self.stop_time,3) + 1.0 / 2.0 * self.a1 * pow(self.stop_time,2) + self.a0 * self.stop_time

    def PlotSpeed(self):
        plt.title('Speed Profile')
        plt.plot(self.t, self.s)



pwd = os.getcwd()
father_path= pwd #os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")
VEL_LIBRARY_PATH = father_path + '/data/jan_vel_matfiles/'
print(VEL_LIBRARY_PATH)
if not os.path.exists(VEL_LIBRARY_PATH):
    os.makedirs(VEL_LIBRARY_PATH)

max_speed = 20
speed_delta = 1


# max_speed = max_speed * 2
# speed_delta = speed_delta * 2
#mode = 'save'
mode = 'draw'
if mode == 'draw':
    speed_list_start = [i*speed_delta for i in range(int(max_speed/speed_delta) + 1)] #8
    # speed_list_start = [5] #8
    speed_list_mid = [i*speed_delta for i in range(int(max_speed/speed_delta) + 1)]
    speed_list_end = [i*speed_delta for i in range(int(max_speed/speed_delta) + 1)]
    # speed_list_end = [4]

    for start_speed in speed_list_start:
        for mid_speed in speed_list_mid:
            for end_speed in speed_list_end:
                SpeedProfile = SpeedParam(v0=start_speed, v1=mid_speed, v2=end_speed, acc0=1)
                if SpeedProfile.illegal == True:
                    continue
                plt.plot(SpeedProfile.t, SpeedProfile.s)
    plt.show()

if mode == 'save':
    speed_list_start = [i*speed_delta for i in range(int(max_speed/speed_delta) + 1)] #8
    speed_list_mid = [i*speed_delta for i in range(int(max_speed/speed_delta) + 1)]
    speed_list_end = [i*speed_delta for i in range(int(max_speed/speed_delta) + 1)]

    for start_speed in speed_list_start:
        speed_dict = {}
        speed_id = 0
        for mid_speed in speed_list_mid:
            for end_speed in speed_list_end:
                SpeedProfile = SpeedParam(v0=start_speed, v1=mid_speed, v2=end_speed, acc0=1)
                if SpeedProfile.illegal == True:
                    continue
                speed_dict[str(speed_id)] = SpeedProfile.vel_list
                speed_id +=1

        mat_name = VEL_LIBRARY_PATH + "vel_init_" + str(start_speed) + "mat.mat"
        io.savemat(mat_name, speed_dict)