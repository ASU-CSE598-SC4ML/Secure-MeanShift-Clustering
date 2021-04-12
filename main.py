#%%
import numpy as np
import torch
from mod2_distcal import distance_calculation
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
import pickle
from check_se_distance import compare_radius


# if __name__ == '__main__':
# Random data generator parameters
n_center = 3
n_point = 50

# Mean-shift Clustering parameters
n_dust = 5
radius = 0.2

# Verify Parameters
Plot_and_GT = True # Show point cloud and ground-truth mean clustering using sklearn
verify_pointenc = False
verify_distcal = False
verify_compare = False

dist_cal1 = distance_calculation(n_point = n_point, n_dust = n_dust, n_center = n_center, radius = radius, if_plot = Plot_and_GT)
dist_cal1.enc(verify_pointenc)
dist_cal1.discal(verify_distcal)
if verify_distcal:
    dist_cal1.verify_discal()
#N: n_dust, M: n_point
#share 0 [N, M], share 1 [N, M]

# share 0 [N, M], share 0 [N, M] -> binary 


# share 0 [N, 1], share 1 [N, 1] -> updated centroid [if any is updated + compare with current centroid]


# share 0 [X, 1], share 1[ X, 1] -> remove repeated.

# -> next iteration 

# -> if none is updated, break.




compare_radius1 = compare_radius(radius = radius)
compare_radius1.compare()
if verify_compare:
    compare_radius1.verify_compare()
'''Expected output:'''
# Ground-Truth Distance is:  0.040053679730797896
# Decrypted Distance is:  tensor(0.0400)
# Ground-Truth Distance is:  0.0
# Decrypted Distance is:  tensor(-1.5259e-05)
# %%
