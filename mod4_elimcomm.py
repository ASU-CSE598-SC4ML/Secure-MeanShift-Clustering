#%%

#Example Usage
# _ , gt_centroid = point_gen([0,1], [0,1], 4, 1000, if_plot = True)
# print("Ground-truth centroid is: ", gt_centroid)

import numpy as np
import numpy as np
import torch
from crypten.mpc.primitives import BinarySharedTensor
import crypten.mpc.primitives.circuit as circuit
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
import pickle


class eliminate_common(object):
    def __init__(self,n_point = 2, n_dust = 1,radius=0.1):
        crypten.init()
        # with open('dist_rank_0.pickle', 'rb') as handle:
        #     temp_dict = pickle.load(handle)
        # temp_dist_enc = temp_dict["distance_share_list_rank0"]
        # self.n_point = len(temp_dist_enc[0])
        # self.n_dust = len(temp_dist_enc)
        # self.radius = radius
        torch.set_num_threads(1)

    @mpc.run_multiprocess(world_size=2)  # Two process will run the identical code below:
    def eliminate(self, verify = False):
        rank = comm.get().get_rank()

        #Receive secret share from compare step.

        # with open('data_rank_{}.pickle'.format(rank), 'rb') as handle:
        #     ss_dict = pickle.load(handle)
        # dust_share_list = ss_dict["dust_share_list_rank{}".format(rank)]
        # point_share_list = ss_dict["point_share_list_rank{}".format(rank)]

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

        #Eliminate Repeated Centroid
        pass

        #Save New Centroid
        return_dict = {}
        return_dict["centroid_share_list_rank{}".format(rank)] = centroid_share_list
        with open('centroid_rank_{}.pickle'.format(rank), 'wb') as handle:
            pickle.dump(return_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    pass
    #Example Usage
    # dist_cal1 = distance_calculation()
    # dist_cal1.enc()
    # dist_cal1.discal()
# %%

