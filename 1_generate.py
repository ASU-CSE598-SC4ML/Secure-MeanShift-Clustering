#%%

#Example Usage
# _ , gt_centroid = point_gen([0,1], [0,1], 4, 1000, if_plot = True)
# print("Ground-truth centroid is: ", gt_centroid)

import numpy as np
import torch
from sklearn.cluster import MeanShift
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def point_gen(range_x, range_y, n_centers, n_points, if_plot = False):
    upper_x = range_x[1]
    lower_x = range_x[0]
    upper_y = range_y[1]
    lower_y = range_y[0]
    center_list = []
    for i in range(n_centers):
        center_list.append([np.random.uniform(low=lower_x, high=upper_x, size=None), np.random.uniform(low=lower_x, high=upper_x, size=None)])
        # print("center {}: ".format(i) + str(center_list[i]))
    n_points_center = n_points // n_centers
    point_list = []
    for j, center in enumerate(center_list):
        for i in range(n_points_center):
            point_list.append(list(np.clip(np.random.multivariate_normal(center, [[0.01,0], [0, 0.01]], size=None), [lower_x, lower_y], [upper_x, upper_y])))
            # print("point {:d}: ".format(j * n_points_center + i) + str(point_list[int(j * n_points_center + i)]))
    
    point_array = np.array(point_list)

    #Plot points being generated
    if if_plot:
        x, y = point_array.T
        colors = cm.rainbow(np.linspace(0, 1, n_centers))
        for i in range(n_centers):
            plt.scatter(x[i*n_points_center:(i+1)*n_points_center], y[i*n_points_center:(i+1)*n_points_center], color=colors[i])
        

    #Call sklearn meanshift to get the ground-truth centroids
    clustering = MeanShift(bandwidth=0.3).fit(point_array)

    gt_centroid = clustering.cluster_centers_
    if if_plot:
        x, y = gt_centroid.T
        plt.scatter(x, y, marker="X", s=256, color = "k")
        plt.show()
    return point_array, gt_centroid

#Example Usage
# _ , gt_centroid = point_gen([0,1], [0,1], 4, 1000, if_plot = True)
# print("Ground-truth centroid is: ", gt_centroid)
# %%
