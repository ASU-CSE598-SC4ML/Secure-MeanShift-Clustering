
#import sys
#sys.path.insert(0, '/home/q/Documents/CrypTen')
import numpy as np
import crypten
import torch

from crypten.mpc.primitives import BinarySharedTensor
import crypten.mpc.primitives.circuit as circuit


def check_le_dist(array1, distance):
    #assume array1 is binaryshared
    result = circuit.le(array1, distance)._tensor
    for i in range(result.size()[0]):
        if result[i]!=0:
            result[i]=1
    return result
    
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