#%%

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
    def __init__(self):
        crypten.init()
        torch.set_num_threads(1)

    @mpc.run_multiprocess(world_size=2)  # Two process will run the identical code below:
    def eliminate(self, verify = False):
        rank = comm.get().get_rank()

        #Receive secret share from compare step.
        with open('centroid_rank_{}.pickle'.format(rank), 'rb') as handle:
            ss_dict = pickle.load(handle)
        old_centroid_share_list = ss_dict["centroid_share_list_rank{}".format(rank)]
        n_dust = len(old_centroid_share_list)
        # print(old_centroid_share_list)
        #Eliminate Repeated Centroid
        centroid_share_list = []
        for i in range(n_dust - 1):
            temp_ss = crypten.cryptensor(0.0, ptype=crypten.ptype.arithmetic)
            for j in range(i+1, n_dust):
                equal_flag = (old_centroid_share_list[i] == old_centroid_share_list[j]).sum()
                temp_ss += equal_flag
            if int(temp_ss.get_plain_text().item()) == 0:
                centroid_share_list.append(old_centroid_share_list[i])
        centroid_share_list.append(old_centroid_share_list[-1])

        if verify:
            if rank == 0:
                print("=========Start of Verification========")
                print("--------Input--------")
            for i in range(n_dust):
                centroid_val = old_centroid_share_list[i].get_plain_text()
                if rank == 0:
                    print("Centroid True Value is: ", centroid_val)
            if rank == 0:
                print("--------Output--------")
            for i, share in enumerate(centroid_share_list):
                result_val = share.get_plain_text()
                if rank == 0:
                    print("Resulted Value is: ", result_val)
            if rank == 0:
                print("=========End of Verification========")

        #Save New Centroid
        return_dict = {}
        return_dict["centroid_share_list_rank{}".format(rank)] = centroid_share_list
        with open('centroid_rank_{}.pickle'.format(rank), 'wb') as handle:
            pickle.dump(return_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# if __name__ == '__main__':
    #Example Usage
    # eliminate_common1 = eliminate_common()
    # eliminate_common1.fake_gen()
    # eliminate_common1.eliminate(True)
# %%

