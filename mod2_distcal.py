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
from os import getpid
import pickle
# import crypten.mpc.primitives.ot.baseOT as baseOT
# from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
import multiprocessing
from multiprocessing import Process, Queue


def sample_dust(point_array, n_dusts):
    sample_idx = np.random.choice(point_array.shape[0], n_dusts, replace=False)
    return point_array[sample_idx]

class distance_calculation(object):
    def __init__(self, n_point = 2, n_dust = 1, n_center = 1):
        crypten.init()
        torch.set_num_threads(1)
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
        self.Q = Queue()

    @mpc.run_multiprocess(world_size=2) # Two process will run the identical code below:
    def enc(self, verify = False):
        dust_share_list = []
        for i in range(self.n_dust):
            dust = list(self.dust_array[i, :])
            dust_enc = crypten.cryptensor(dust, ptype=crypten.ptype.arithmetic)
            dust_share_list.append(dust_enc)
        point_share_list = []
        for i in range(self.n_point):
            point =  list(self.point_array[i, :])
            point_enc = crypten.cryptensor(point, ptype=crypten.ptype.arithmetic)
            point_share_list.append(point_enc)
        if verify:
            print("=========Start of Verification========")
            for i in range(self.n_dust):
                print("Dust to be Encrypted is: ", self.dust_array[i, :])
                print("Decrypted Dust is: ", dust_share_list[i].get_plain_text())
            for i in range(self.n_point):
                print("Point to be Encrypted is: ", self.point_array[i, :])
                print("Decrypted Point is: ", point_share_list[i].get_plain_text())
            print("=========End of Verification========")
        # return dust_share_list, point_share_list, save secret share to file.
        rank = comm.get().get_rank()
        return_dict = {}
        return_dict["dust_share_list_rank{}".format(rank)] = dust_share_list
        return_dict["point_share_list_rank{}".format(rank)] = point_share_list
        with open('rank_{}.pickle'.format(rank), 'wb') as handle:
            pickle.dump(return_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @mpc.run_multiprocess(world_size=2)  # Two process will run the identical code below:
    def discal(self, verify = False):
        rank = comm.get().get_rank()

        #Receive secret share from enc step.
        with open('rank_{}.pickle'.format(rank), 'rb') as handle:
            ss_dict = pickle.load(handle)
        dust_share_list = ss_dict["dust_share_list_rank{}".format(rank)]
        point_share_list = ss_dict["point_share_list_rank{}".format(rank)]
        if verify:
            print("=========Start of Verification========")
            for i in range(self.n_dust):
                print("Dust to be Encrypted is: ", self.dust_array[i, :])
                print("Decrypted Dust is: ", dust_share_list[i].get_plain_text())
            for i in range(self.n_point):
                print("Point to be Encrypted is: ", self.point_array[i, :])
                print("Decrypted Point is: ", point_share_list[i].get_plain_text())
            print("=========End of Verification========")
        
        # Calculate the distance using the secret share.
        

if __name__ == '__main__':
    dist_cal1 = distance_calculation()
    dist_cal1.enc()
    dist_cal1.discal(True)
# %%

