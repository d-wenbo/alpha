from re import S
import numpy as np
import cv2
import random
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import csv
import pandas as pd

def search_list(list,num):
    n = 0
    for i in list:
        if i >= num:
            n += 1
        else:
            continue
    return n


args = sys.argv
'''
filename = 'alpha_AREA07.csv'
output_file = 'test.csv'
'''


filename = args[1]
output_file = args[2]

df_m = pd.read_csv(filename,sep=',')

f = open(output_file, 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(['img_name','angle','score','x_label1','y_label1'])



list_num_point_rm = []


if __name__ == "__main__":


    dict_filename_number = {}
    dict_for_label1 = {}
    list_imgname_t = []
    for j in range(df_m.shape[0]):
        label = int(df_m.loc[j][0])
        score = float(df_m.loc[j][1])
        x_left = int(df_m.loc[j][2])
        y_left = int(df_m.loc[j][3])
        x_right = int(df_m.loc[j][4])
        y_right = int(df_m.loc[j][5])
        img_name_df = df_m.loc[j][6]
        
        if not img_name_df in list_imgname_t:
            list_imgname_t.append(img_name_df)
        
        this_info = {
                    "label": label,
                    "score": score,
                    "x": int((x_left + x_right)/2),
                    "y": int((y_left + y_right)/2),
                    "flag_ghost": False,
                    "img_name_df": img_name_df
                    }

        if label ==1:
            if not img_name_df in dict_for_label1:
                filename_number = {img_name_df: {"count":0, "info":[]}}
                dict_for_label1.update(filename_number)
            
            dict_for_label1[img_name_df]["count"] += 1
            dict_for_label1[img_name_df]["info"].append(this_info)

        else:
            #increse img_name                                                   
            if not img_name_df in dict_filename_number:
                filename_number = {img_name_df: {"count":0, "info":[]}}
                dict_filename_number.update(filename_number)
            
            # increse infomation                                                
            dict_filename_number[img_name_df]["count"] += 1
            dict_filename_number[img_name_df]["info"].append(this_info)

    
    # ghost reduction                                                           
    list_distance = []
    threshold_distance = 6
    '''
    for key, val in dict_filename_number.items():
        list_info = val["info"]
        num_duplicate = 0
        num_points = 0
        num_points = len(list_info)
        for i in range(len(list_info)):
            if list_info[i]["flag_ghost"] == True:

                continue
            for j in range(i + 1, len(list_info)):


                dx = list_info[i]["x"] - list_info[j]["x"]
                dy = list_info[i]["y"] - list_info[j]["y"]
                dist = np.hypot(dx, dy)
                list_distance.append(dist)
                if dist < threshold_distance:
                    list_info[j]["flag_ghost"] = True
                    num_duplicate +=1
        list_num_point_rm.append(num_points-num_duplicate)
    #print(list_num_point_rm)
    n = search_list(list_num_point_rm,5)
    #print(n)                                                                   
    '''
    
    



    # drawing                                                                   
    list_num_point = []
    for i in range(len(list_imgname_t)):
        
        img_name = list_imgname_t[i]
        list_x = []
        list_y = []
        list_score = []
        list_x_label1 =[]
        list_y_label1 =[]
        list_score_label1 = []
        list_angle = []
        list_score_angle = []
        if not img_name in dict_filename_number:
            continue
        list_info =  dict_filename_number[img_name]["info"]
        for info in list_info:
            x = info["x"]
            y = info["y"]
            score = info["score"]
            list_x.append(x)
            list_y.append(y)
            list_score.append(score)
            
            
        list_score.reverse()
        list_x.reverse()
        list_y.reverse()


        if img_name in dict_for_label1:

            list_info_label1 = dict_for_label1[img_name]["info"]
            #print(list_info_label1)
            for info_1 in list_info_label1:
                x = info_1["x"]
                y = info_1["y"]
                score = info_1["score"]
                list_x_label1.append(x)
                list_y_label1.append(y)
                list_score_label1.append(score)

            list_x_label1_base = []
            list_y_label1_base = []
            list_score_label1_base = []
            index = list_score_label1.index(max(list_score_label1))
            x_label1_base = list_x_label1[index]
            y_label1_base = list_y_label1[index]
            score_label1_base = list_score_label1[index]
            list_x_label1_base.append(x_label1_base)
            list_y_label1_base.append(y_label1_base)
            list_score_label1_base.append(score_label1_base)
            
        if len(list_score_label1) != 0 and len(list_score) != 0:

            

            
            
            for info in list_info:
                x = info["x"]
                y = info["y"]
                score = info["score"]
                calc_x = x - x_label1_base
                calc_y = y - y_label1_base
                angle =  math.degrees(math.atan2(calc_y,calc_x))
                
                writer.writerow([img_name,angle,score,x_label1_base,y_label1_base])
                list_angle.append(angle)
                list_score_angle.append(score)
            
            
            
            
        

        # histogram                                                                 

    

