import numpy as np
import random
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
from scipy.stats import norm
from scipy.optimize import curve_fit
import pickle
import pandas as pd
import cv2
import math

args = sys.argv



pickelefile = args[1]
picklefile_write = args[2]
txtfile = args[3]
threshold = float(args[4])


t = open(txtfile, 'w')

with open(pickelefile, mode='rb') as f:
    dict_cluster = pickle.load(f)
    
plt.rcParams['font.size'] = 14

def func(x, a, mu, sigma,c):
    
    return a * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2 )) + c
def get_histogram_arrays(list_val, n_bin, x_min, x_max):
    bin_heights, bin_borders = np.histogram(list_val, n_bin, (x_min, x_max))
    bin_middles = 0.5*(bin_borders[1:] + bin_borders[:-1])
    return bin_middles, bin_heights
def line_equ(x,a,b):
    y = a*x +b
    return y



if __name__ == "__main__":   
   
    
    
    
    
    
    for name in dict_cluster:
        
        list_angle_cluster_d = dict_cluster[name]["angle"]
        list_score_cluster_d = dict_cluster[name]["score"]
        
        
        

        
    num_detected = []
    
    line_min = 2
    line_max = 6
    num = 1
    list_score_all = []
    for name in dict_cluster:
        
        list_angle_cluster_d = dict_cluster[name]["angle"]
        list_score_cluster_d = dict_cluster[name]["score"]
        list_score_all.extend(list_score_cluster_d)
        
        arr_score_cluster_d = np.array(list_score_cluster_d)
        #print(arr_score_cluster_d.size)
        arr_score_cluster_d_rm= np.delete(arr_score_cluster_d,np.where(arr_score_cluster_d<threshold))
        num_detected.append(arr_score_cluster_d_rm.size)
        #print(arr_score_cluster_d_rm.size)
        if line_min <= arr_score_cluster_d_rm.size <= line_max :
            print(num,name)
            t.write(name + '\n')
            num +=1
    print(num_detected)
    print(len(num_detected))
    #print(list_score_all)
    with open(picklefile_write,'wb') as n:
        pickle.dump(num_detected, n)
    with open('not_vertex_score.pickle','wb') as g:
        pickle.dump(list_score_all, g)    

        
        
    
    
    
        
