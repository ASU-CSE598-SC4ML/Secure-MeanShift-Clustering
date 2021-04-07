#%%

#Example Usage
# _ , gt_centroid = point_gen([0,1], [0,1], 4, 1000, if_plot = True)
# print("Ground-truth centroid is: ", gt_centroid)

import numpy as np
import torch
from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mod1_generate import point_gen
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
import crypten.mpc.primitives.ot.baseOT as baseOT
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element


def sample_dust(point_array, n_dusts):
    sample_idx = np.random.choice(point_array.shape[0], n_dusts, replace=False)
    return point_array[sample_idx]

class distance_calculation(object):
    def __init__(self, n_point = 2, n_dust = 1, n_center = 1):
        
        if_plot = True
        self.upper_x = 1
        self.lower_x = 0
        self.upper_y = 1
        self.lower_y = 0
        self.n_point = n_point
        self.n_dust = n_dust
        self.n_center = n_center
        self.point_array, self.gt_centroid = point_gen([self.lower_x,self.upper_x], [self.lower_y,self.upper_y], self.n_center, self.n_point, if_plot = if_plot)
        self.dust_array = sample_dust(self.point_array, self.n_dust)
        if if_plot:
            x, y = self.dust_array.T
            plt.scatter(x, y, marker="*", s=256, color = "m")
            plt.xlim(self.lower_x,  self.upper_x)
            plt.ylim(self.lower_y,  self.upper_y)
            plt.show()

    @mpc.run_multiprocess(world_size=2)
    def fit(self):
        #Should return a [n_dust, n_point] array for each of N server (N = 2) 
        res_share_list = [] #share for server 0 of the [n_dust, n_point] array
        for i in range(self.n_dust):
            dist_share_list = []
            dust = list(self.dust_array[i, :])
            for j in range(self.n_point):
                point =  list(self.point_array[j, :])
                point_enc = crypten.cryptensor(point, ptype=crypten.mpc.arithmetic)
                dust_enc = crypten.cryptensor(dust, ptype=crypten.mpc.arithmetic)
                dist_share_list.append(point_enc)
                # x_enc = crypten.cryptensor([2, 3], ptype=crypten.mpc.binary)
        res_share_list.append(dist_share_list)
        rank = comm.get().get_rank()
        crypten.print(f"\nRank {rank}:\n {str(res_share_list)}\n")
                # dist_share0, dist_share1 = [0.1, 0.09]
                # dist_share0_list.append(dist_share0)
                # dist_share1_list.append(dist_share1)
            # res_share_list.append(dist_share_list)
        # return np.array(res_share0_list), np.array(res_share1_list)
    # pass

@mpc.run_multiprocess(world_size=2)
def examine_arithmetic_shares():
    x_enc = crypten.cryptensor([1, 2, 3], ptype=crypten.mpc.arithmetic)
    
    rank = comm.get().get_rank()
    crypten.print(f"\nRank {rank}:\n {x_enc}\n")

if __name__ == '__main__':

    #initialize crypten
    # OMP_NUM_THREADS=2
    #Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
    crypten.init()
    torch.set_num_threads(1)
    dist_cal1 = distance_calculation()
    # arr1, arr2 = dist_cal1.fit()
    dist_cal1.fit()
    # print(arr1.shape, arr2.shape)
    # print(torch.get_num_threads())
    
    x = examine_arithmetic_shares()

    # ot = baseOT.BaseOT((self.rank + 1) % self.world_size)
    # msg0s = ["9"]
    # msg1s = ["23"]
    # ot.send(msg0s, msg1s)
    # choices = [1]
    
    # msgcs = ot.receive(choices)
    # print(msgcs)
# %%

