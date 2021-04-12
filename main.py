import numpy as np
import torch
from mod2_distcal import distance_calculation
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
import pickle



if __name__ == '__main__':
    dist_cal1 = distance_calculation(n_point = 2, n_dust = 1, n_center = 1)
    dist_cal1.enc()
    dist_cal1.discal()
    #N: n_dust, M: n_point
    #share 0 [N, M], share 1 [N, M]

    # share 0 [N, M], share 0 [N, M] -> binary 


    # share 0 [N, 1], share 1 [N, 1] -> updated centroid [if any is updated + compare with current centroid]


    # share 0 [X, 1], share 1[ X, 1] -> remove repeated.

    # -> next iteration 

    # -> if none is updated, break.


    verify = True
    if verify:
        dist_cal1.verify_discal()

    '''Expected output:'''
    # Ground-Truth Distance is:  0.040053679730797896
    # Decrypted Distance is:  tensor(0.0400)
    # Ground-Truth Distance is:  0.0
    # Decrypted Distance is:  tensor(-1.5259e-05)