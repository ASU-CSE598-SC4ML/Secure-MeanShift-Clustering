import numpy as np
import torch
from mod2_distcal import distance_calculation
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
import pickle



if __name__ == '__main__':
    dist_cal1 = distance_calculation()
    dist_cal1.enc()
    dist_cal1.discal()
    verify = True
    if verify:
        dist_cal1.verify_discal()