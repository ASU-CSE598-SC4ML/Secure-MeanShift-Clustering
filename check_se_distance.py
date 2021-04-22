
#import sys
#sys.path.insert(0, '/home/q/Documents/CrypTen')
import numpy as np
import torch
from crypten.mpc.primitives import BinarySharedTensor
import crypten.mpc.primitives.circuit as circuit
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
import pickle

class compare_radius(object):
    def __init__(self,radius=0.1):
        crypten.init()
        with open('dist_rank_0.pickle', 'rb') as handle:
            temp_dict = pickle.load(handle)
        temp_dist_enc = temp_dict["distance_share_list_rank0"]
        self.n_point = len(temp_dist_enc[0])
        self.n_dust = len(temp_dist_enc)
        self.radius = radius
        torch.set_num_threads(1)

    @mpc.run_multiprocess(world_size=2)
    def compare(self, debug=True):
        rank = comm.get().get_rank()
        #Receive secret share from previous function.
        with open('dist_rank_{}.pickle'.format(rank), 'rb') as handle:
            dist_dict = pickle.load(handle)
        dist_enc = dist_dict["distance_share_list_rank{}".format(rank)]
        dist_enc_new = crypten.cryptensor(torch.ones(self.n_dust, self.n_point))
        for i in range(self.n_dust):
            for j in range(self.n_point):
                dist_enc_new[i,j]=dist_enc[i][j]

        with open('data_rank_{}.pickle'.format(rank), 'rb') as handle:
            ss_dict = pickle.load(handle)
        point_enc = ss_dict["point_share_list_rank{}".format(rank)]
        point_enc_new = crypten.cryptensor(torch.ones(self.n_point, 2))
        for i in range(self.n_point):
            point_enc_new[i,:]=point_enc[i][:]

        with open('centroid_rank_{}.pickle'.format(rank), 'rb') as handle:
            ss_dict = pickle.load(handle)
        dust_enc = ss_dict["centroid_share_list_rank{}".format(rank)]
        dust_enc_new = crypten.cryptensor(torch.ones(self.n_dust, 2))
        for i in range(self.n_dust):
            dust_enc_new[i,:]=dust_enc[i][:]

        #temp is the radius
        distance_bool_list = []
        changed = False
        for i in range(self.n_dust):
            templist = []
            temprad = torch.ones(len(dist_enc[i]))*self.radius
            
            #create shared radius ([r,r,r,r....])
            radius_enc = crypten.cryptensor(temprad, ptype=crypten.ptype.arithmetic)
            
            #calculates if point distance is le radius
            temp_bool = crypten.cryptensor(temprad, ptype=crypten.ptype.arithmetic)
            temp_bool = dist_enc_new[i]<=radius_enc
            
            #multiply point value with 0/1 matrix, 
            #result should be the updated centroid location
            temp_pts = crypten.cryptensor(torch.ones(point_enc_new.shape))
            for j in range(self.n_point):
                temp_pts[j,:] = point_enc_new[j,:]*temp_bool[j]
            
            #sum them up, divide by sum of 0/1 matrix
            updated_centroid = (temp_pts.sum(0))/temp_bool.sum()
            """if debug:
                a = updated_centroid.get_plain_text()
                b = dust_enc_new[i,:].get_plain_text()
                if rank==0:
                    print("a and b:")
                    print(a)
                    print(b)"""
            #get plain text of ne, if is 1(not equal) then set changed
            changetemp = (updated_centroid!=dust_enc_new[i,:]).get_plain_text().sum().item()
            if (changetemp):
                #if changed, set flag and change dust
                changed = True
                if debug:
                    a = updated_centroid.get_plain_text()
                    b = dust_enc_new[i,:].get_plain_text()
                    if rank==0:
                        print("a and b:")
                        print(a)
                        print(b)
                dust_enc[i] = updated_centroid
        
        #if changed, then update the data file with new dust
        if changed:
            ss_dict["centroid_share_list_rank{}".format(rank)]= dust_enc
            with open('centroid_rank_{}.pickle'.format(rank), 'wb') as handle:
                pickle.dump(ss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('result.pickle', 'wb') as handle:
                pickle.dump(changed, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #The verify is still incorrect, I'll fix it and test if the code works correctly
    @mpc.run_multiprocess(world_size=2)
    def verify_compare(self):
        rank = comm.get().get_rank()

        #Receive secret share from distcal step.
        with open('compare_results_{}.pickle'.format(rank), 'rb') as handle:
            dist_dict = pickle.load(handle)
        results_share_list = dist_dict["distance_results_rank{}".format(rank)]


        with open('dist_rank_{}.pickle'.format(rank), 'rb') as handle:
            dist_dict = pickle.load(handle)
        dist_enc = dist_dict["distance_share_list_rank{}".format(rank)]

        #Verify each distance
        if rank == 0:
            print("=========Start of Verification========")
        for i in range(self.n_dust):
            for j in range(self.n_point):
                gt_dist = dist_enc[i][j]
                radius_tensor = torch.ones(gt_dist.shape)*self.radius
                gt_calculated = (gt_dist<=radius_tensor).get_plain_text()

                if rank == 0:
                    print("Ground-Truth is: not implemented")
                decrypted = results_share_list[i][j].get_plain_text()
                if rank == 0:
                    print("Decrypted is: not implemented")
        if rank == 0:
            print("=========End of Verification========")

#with open('compare_results_1.pickle', 'rb') as handle:
#    a = pickle.load(handle)
#    print(a)
"""
def check_le_dist(array1, distance):
    #assume array1 is binaryshared
    result = circuit.le(array1, distance)._tensor
    for i in range(result.size()[0]):
        if result[i]!=0:
            result[i]=1
    return result
"""  
"""
def test_function():
    dist=3
    testarray = torch.tensor([1,2,3,4,5])
    enctest = BinarySharedTensor(testarray)
    print(testarray)
    distance_tensor = torch.tensor([3,3,3,3,3])
    print(distance_tensor)
    encrypted_distance = BinarySharedTensor(distance_tensor)
    result = check_le_dist(enctest, encrypted_distance)

    print(result)

crypten.init()
test_function()
"""