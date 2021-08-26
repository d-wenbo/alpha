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

filename = args[1]

pickelefile = args[2]
picklefile_write = args[3]
threshold = float(args[4])


df = pd.read_csv(filename,sep=',')

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
   
    
    dict_co_label1 = {}
    for i in range(df.shape[0]):
        img_name = df.loc[i][0]
        x_label1 = df.loc[i][3]
        y_label1 = df.loc[i][4]
        
        if not img_name in dict_co_label1:
            info = {img_name:{"x_label1":[],"y_label1":[]}}
            dict_co_label1.update(info)
        dict_co_label1[img_name]["x_label1"].append(x_label1)
        dict_co_label1[img_name]["y_label1"].append(y_label1)
    #print(dict_co_label1)
    
    
    for name in dict_cluster:
        
        list_angle_cluster_d = dict_cluster[name]["angle"]
        list_score_cluster_d = dict_cluster[name]["score"]
        list_x_label1 = dict_co_label1[name]['x_label1']
        list_y_label1 = dict_co_label1[name]['y_label1']
        list_x = []
        list_y = []
        #img = cv2.imread(img_file + '/'+ str(name), 0)
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        x_label1 = list_x_label1[0]
        y_label1 = list_y_label1[0]

        for angle_cluster_d,score_cluster_d in zip(list_angle_cluster_d,list_score_cluster_d):
            
            theta = math.radians(angle_cluster_d)
            x = int(x_label1 + 100 * math.cos(theta))
            y = int(y_label1 + 100 * math.sin(theta))
            list_x.append(x)
            list_y.append(y)

            
                
        
        fig = plt.figure()
        


    
    
    
    
    num_detected = []
    
    
    
    list_count_diff = [] 
    for name in dict_cluster:
        
        list_angle_cluster_d = dict_cluster[name]["angle"]
        list_score_cluster_d = dict_cluster[name]["score"]
        arr_score_cluster_d = np.array(list_score_cluster_d)
        #print(arr_score_cluster_d.size)
        arr_score_cluster_d_rm= np.delete(arr_score_cluster_d,np.where(arr_score_cluster_d<threshold))
        num_detected.append(arr_score_cluster_d_rm.size)
        #print(arr_score_cluster_d_rm.size)
    
    print(num_detected)
    print(len(num_detected))
    
    with open(picklefile_write,'wb') as n:
        pickle.dump(num_detected, n)

        
        
    
    
    
        
