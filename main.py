#%%
import numpy as np
import torch
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
import pickle
from mod1_generate import point_gen
from mod2_distcal import distance_calculation
from mod3_compare import compare_radius
from mod4_elimcomm import eliminate_common
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
import time
import argparse
import logging
import sys
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
from datetime import datetime
def setup_logger(name, log_file, level=logging.INFO, console_out = False):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    if console_out:
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)
    return logger

class meanshift(object):
    def __init__(self, point_array, radius = 0.1, n_dust = 8, verify = False):
        self.point_array = point_array
        self.n_point = point_array.shape[0]
        self.radius = radius # Mean-shift Clustering parameters
        self.n_dust = n_dust # Dust Heuristic parameters, also used to track the number of centroid
        self.Plot_and_GT = True # Show point cloud and ground-truth mean clustering using sklearn
        self.verify_pointenc = verify # Show point cloud is correctly encrypted
        self.verify_distcal = verify # Show distance calculation is correct
        self.verify_compare = verify # Show distance compare is correct

    def fit(self):

        dist_cal1 = distance_calculation(self.point_array, n_dust = self.n_dust, radius = self.radius, if_plot = self.Plot_and_GT)
        
        compare_radius1 = compare_radius(self.radius, self.n_point)
        eliminate_common1 = eliminate_common()
        dist_cal1.enc(self.verify_pointenc)
        i = 0
        max_iter = 20
        iter_time_list = []
        while (i < max_iter):
            
            dist_cal1.print_centroid()
            start_time = time.time()
            # start by dist cal with the current n_dust
            dist_cal1.discal(self.verify_distcal)
            dist_cal_time = time.time() - start_time
            compare_radius1.compare()
            compare_radius_time = time.time() - start_time - dist_cal_time
            with open('result.pickle', 'rb') as handle:
                if_exit = pickle.load(handle)

            if not if_exit:
                break
            
            eliminate_common1.eliminate()
            #update n_dust
            iter_time = time.time() - start_time
            print("iter {} - time cost is {:1.4f}s (dist: {:1.4f}s/compare: {:1.4f}s)".format(i, iter_time, dist_cal_time, compare_radius_time))
            iter_time_list.append(iter_time)
            i += 1
        dist_cal1.get_plain_centroid()
        with open('plain_centroid.pickle', 'rb') as handle:
            plain_centroid = np.asarray(pickle.load(handle))
        

        return plain_centroid, iter_time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_dust', type=int, default=8)
    parser.add_argument('--n_point', type=int, default=100)
    parser.add_argument('--n_centers', type=int, default=8)
    parser.add_argument('--radius', type=float, default=0.1)
    args = parser.parse_args()
    n_point = args.n_point
    n_dust = args.n_dust
    radius = args.radius
    log_file = 'test_point{}_dust{}_radius{}_center{}_{}.log'.format(n_point, n_dust, radius, args.n_centers, datetime.today().strftime('%m%d%H%M'))
    logger = setup_logger('test_logger', log_file, level = logging.DEBUG, console_out = True)
    point_array = point_gen([0.0,1.0], [0.0,1.0], n_centers = args.n_centers, n_points = n_point, radius = radius, if_plot = True)
    ms = meanshift(point_array, radius, n_dust)
    plain_centroid, iter_time_list = ms.fit()
    logger.debug("Overall time cost is {:1.4f}s for total of {} iterations!".format(sum(iter_time_list), len(iter_time_list)))
    logger.debug("Average time cost per iteration is {:1.4f}s!".format(sum(iter_time_list)/len(iter_time_list)))

    #Call sklearn meanshift to get the ground-truth centroids
    clustering = MeanShift(bandwidth=0.2).fit(point_array)

    #Plot Ground-Truth Clustering Center
    gt_centroid = clustering.cluster_centers_

    logger.debug("Final centroid is " + str(plain_centroid))
    logger.debug("Ground Truth centroid is " + str(gt_centroid))

    x, y = gt_centroid.T
    plt.scatter(x, y, marker="X", s=128, color = "k") #balck x is the ground-truth
    x, y = plain_centroid.T
    plt.scatter(x, y, marker="X", s=128, color = "b") #blue is ours
    plt.xlim(0.0,  1.0)
    plt.ylim(0.0,  1.0)
    plt.show()
    plt.savefig('cluster_results_{}.png'.format(datetime.today().strftime('%m%d%H%M')))

    # plot
# %%
