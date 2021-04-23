#%%
import numpy as np
import torch
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
import pickle
from mod2_distcal import distance_calculation
from mod3_compare import compare_radius
from mod4_elimcomm import eliminate_common
from matplotlib import pyplot as plt
class meanshift(object):
    def __init__(self, n_point, radius, n_center = 5, n_dust = 10, verify = False):
        self.n_point = n_point # Random data generator parameters
        self.radius = radius # Mean-shift Clustering parameters
        self.n_center = n_center # Random data generator parameters
        self.n_dust = n_dust # Dust Heuristic parameters, also used to track the number of centroid
        self.Plot_and_GT = True # Show point cloud and ground-truth mean clustering using sklearn
        self.verify_pointenc = verify # Show point cloud is correctly encrypted
        self.verify_distcal = verify # Show distance calculation is correct
        self.verify_compare = verify # Show distance compare is correct
    def fit(self):
        dist_cal1 = distance_calculation(n_point = self.n_point, n_dust = self.n_dust, 
                        n_center = self.n_center, radius = self.radius, if_plot = self.Plot_and_GT)
        
        compare_radius1 = compare_radius(self.radius, self.n_point)
        eliminate_common1 = eliminate_common()
        dist_cal1.enc(self.verify_pointenc)
        i = 0
        max_iter = 20
        while (i < max_iter):
            # start by dist cal with the current n_dust
            dist_cal1.print_centroid()

            dist_cal1.discal(self.verify_distcal)
            
            compare_radius1.compare()
            
            with open('result.pickle', 'rb') as handle:
                if_exit = pickle.load(handle)

            if not if_exit:
                break
            else:
                print("centroids are updated!")
            
            eliminate_common1.eliminate()
            #update n_dust
            i += 1

        dist_cal1.get_plain_centroid()
        with open('plain_centroid.pickle', 'rb') as handle:
            plain_centroid = np.asarray(pickle.load(handle))
        print("Final centroid is ", plain_centroid)
        print("Ground Truth centroid is ", dist_cal1.gt_centroid)
        x, y = dist_cal1.gt_centroid.T
        plt.scatter(x, y, marker="X", s=128, color = "k")
        x, y = plain_centroid.T
        plt.scatter(x, y, marker="X", s=128, color = "b")
        plt.xlim(0.0,  1.0)
        plt.ylim(0.0,  1.0)
        plt.show()
        # plot
n_point = 100
radius = 0.1
ms = meanshift(n_point, radius)
ms.fit()
# %%

# if __name__ == '__main__':