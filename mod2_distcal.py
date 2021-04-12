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
import pickle

#Example Usage
    # dist_cal1 = distance_calculation(2, 1, 1)
    # dist_cal1.enc()
    # dist_cal1.discal()

def sample_dust(point_array, n_dusts):
    sample_idx = np.random.choice(point_array.shape[0], n_dusts, replace=False)
    return point_array[sample_idx]

class distance_calculation(object):
    def __init__(self, n_point = 2, n_dust = 1, n_center = 1, radius = 0.1, if_plot = True):
        crypten.init()
        torch.set_num_threads(1)
        self.upper_x = 1
        self.lower_x = 0
        self.upper_y = 1
        self.lower_y = 0
        self.n_point = n_point
        self.n_dust = n_dust
        self.n_center = n_center
        self.radius = radius
        self.point_array, self.gt_centroid = point_gen([self.lower_x,self.upper_x], [self.lower_y,self.upper_y], self.n_center, self.n_point, radius = self.radius, if_plot = if_plot)
        self.dust_array = sample_dust(self.point_array, self.n_dust)
        if if_plot:
            x, y = self.dust_array.T
            plt.scatter(x, y, marker="*", s=256, color = "m")
            plt.xlim(self.lower_x,  self.upper_x)
            plt.ylim(self.lower_y,  self.upper_y)
            plt.show()

    @mpc.run_multiprocess(world_size=2) # Two process will run the identical code below:
    def enc(self, verify = False):
        rank = comm.get().get_rank()
        dust_share_list = []
        print(self.dust_array.shape, self.point_array.shape)
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
            if rank == 0:
                print("=========Start of Verification========")
            for i in range(self.n_dust):
                dust_val = dust_share_list[i].get_plain_text()
                if rank == 0:
                    print("Dust to be Encrypted is: ", self.dust_array[i, :])
                    print("Decrypted Dust is: ", dust_val)
            for i in range(self.n_point):
                point_val = point_share_list[i].get_plain_text()
                if rank == 0:
                    print("Point to be Encrypted is: ", self.point_array[i, :])
                    print("Decrypted Point is: ", point_val)
            if rank == 0:
                print("=========End of Verification========")

        # return dust_share_list, point_share_list, save secret share to file
        return_dict1 = {}
        return_dict1["point_share_list_rank{}".format(rank)] = point_share_list
        with open('data_rank_{}.pickle'.format(rank), 'wb') as handle:
            pickle.dump(return_dict1, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return_dict2 = {}
        return_dict2["centroid_share_list_rank{}".format(rank)] = dust_share_list
        with open('centroid_rank_{}.pickle'.format(rank), 'wb') as handle:
            pickle.dump(return_dict2, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @mpc.run_multiprocess(world_size=2)  # Two process will run the identical code below:
    def discal(self, verify = False):
        rank = comm.get().get_rank()

        #Receive secret share from enc step.
        with open('data_rank_{}.pickle'.format(rank), 'rb') as handle:
            ss_dict = pickle.load(handle)
        point_share_list = ss_dict["point_share_list_rank{}".format(rank)]

        with open('centroid_rank_{}.pickle'.format(rank), 'rb') as handle:
            ss_dict = pickle.load(handle)
        centroid_share_list = ss_dict["centroid_share_list_rank{}".format(rank)]
        
        # Verify the correctness of received shares.
        if verify:
            if rank == 0:
                print("=========Start of Verification========")
            for i in range(self.n_dust):
                dust_val = centroid_share_list[i].get_plain_text()
                if rank == 0:
                    print("Dust to be Encrypted is: ", self.dust_array[i, :])
                    print("Decrypted Dust is: ", dust_val)
            for i in range(self.n_point):
                point_val = point_share_list[i].get_plain_text()
                if rank == 0:
                    print("Point to be Encrypted is: ", self.point_array[i, :])
                    print("Decrypted Point is: ", point_val)
            if rank == 0:
                print("=========End of Verification========")
        
        # Verify distance calculation the secret share.
        if verify:
            if rank == 0:
                print("=========Start of Verification========")
            true_distance = sum(((point_share_list[0].get_plain_text() - point_share_list[1].get_plain_text())**2))
            dist_sum = (point_share_list[0] ** 2 +  point_share_list[1] ** 2 - 2 * (point_share_list[0] * point_share_list[1])).sum()
            dist_sum_val = dist_sum.get_plain_text()
            if rank == 0:
                print("Ground-Truth Distance is: ", true_distance)
                print("Decrypted Distance is: ", dist_sum_val)
                print("=========End of Verification========")

        #Calculate Distance
        distance_share_list = []
        for i in range(self.n_dust):
            tmp_list = []
            for j in range(self.n_point):
                dist_sum = (centroid_share_list[i] ** 2 +  point_share_list[j] ** 2 - 2 * (centroid_share_list[i] * point_share_list[j])).sum()
                tmp_list.append(dist_sum)
            distance_share_list.append(tmp_list)

        #Save Distance
        return_dict = {}
        return_dict["distance_share_list_rank{}".format(rank)] = distance_share_list
        with open('dist_rank_{}.pickle'.format(rank), 'wb') as handle:
            pickle.dump(return_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @mpc.run_multiprocess(world_size=2)  # Two process will run the identical code below:
    def verify_discal(self):
        rank = comm.get().get_rank()

        #Receive secret share from distcal step.
        with open('dist_rank_{}.pickle'.format(rank), 'rb') as handle:
            dist_dict = pickle.load(handle)
        distance_share_list = dist_dict["distance_share_list_rank{}".format(rank)]


        #Verify each distance
        if rank == 0:
            print("=========Start of Verification========")
        for i in range(self.n_dust):
            for j in range(self.n_point):
                if rank == 0:
                    gt_dist = sum((self.dust_array[i, :] - self.point_array[j, :]) ** 2)
                    print("Ground-Truth Distance is: ", gt_dist)
                dist_ss = distance_share_list[i][j].get_plain_text()
                if rank == 0:
                    print("Decrypted Distance is: ", dist_ss)
        if rank == 0:
            print("=========End of Verification========")
            
if __name__ == '__main__':
    pass
    #Example Usage
    # dist_cal1 = distance_calculation()
    # dist_cal1.enc()
    # dist_cal1.discal()
# %%

