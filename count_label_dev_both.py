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
filename = args[1]
img_file = args[2]
outputname = args[3]
output_file = args[4]
df_m = pd.read_csv(filename,sep=',')

f = open(output_file, 'w')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(['img_name','angle','score','x_label1','y_label1'])


new_dir_path = outputname
os.makedirs(new_dir_path,exist_ok = True)
new_dir_path_graph = new_dir_path + '/' + 'hist/'
os.makedirs(new_dir_path_graph,exist_ok = True)
new_dir_path_img = new_dir_path + '/' +'img_putted/'
os.makedirs(new_dir_path_img,exist_ok = True)
new_dir_path_graph_colorbar = new_dir_path + '/' + 'hist_colorbar/'
os.makedirs(new_dir_path_graph_colorbar,exist_ok=True)
new_dir_path_graph_angle = new_dir_path + '/' + 'hist_angle/'
os.makedirs(new_dir_path_graph_angle,exist_ok = True)

list_imgname = glob.glob(img_file + "/*.png")
list_imgname.sort()


list_num_point_rm = []


if __name__ == "__main__":


    dict_filename_number = {}
    dict_for_label1 = {}
    for j in range(df_m.shape[0]):
        label = int(df_m.loc[j][0])
        score = float(df_m.loc[j][1])
        x_left = int(df_m.loc[j][2])
        y_left = int(df_m.loc[j][3])
        x_right = int(df_m.loc[j][4])
        y_right = int(df_m.loc[j][5])
        img_name_df = df_m.loc[j][6]
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

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(list_distance, range=(0,50), bins = 50) # ,align ='left' )          

    ax.set_title("distribution of distance between detected point",fontsize=12)
    ax.set_xlabel("distance [pixel]", size = 10)
    ax.set_ylabel("", size = 10)
    #plt.xticks(np.arange(0, 20 + 1, 2 ))                                       
    plt.tight_layout()
    fig.savefig(new_dir_path_graph + 'hist_distance.png')
    plt.clf()



    # drawing                                                                   
    list_num_point = []
    for i in range(len(list_imgname)):
        img = cv2.imread(list_imgname[i], 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_convert = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_angle = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_name = os.path.basename(list_imgname[i])
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
            if info["flag_ghost"] == True:
                this_color = (255, 0, 0)
            else:
                this_color = (0, 0, 255)
            cv2.circle(img, (x, y), 5, this_color, -1)
            cv2.putText(img, str(score), (x , y), cv2.FONT_HERSHEY_PLAIN, 2, this_color, 1, cv2.LINE_AA)

        list_score.reverse()
        list_x.reverse()
        list_y.reverse()


        if img_name in dict_for_label1:

            list_info_label1 = dict_for_label1[img_name]["info"]
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

            fig = plt.figure()

            plt.scatter(list_x,list_y,s = 10,c = list_score,cmap ='inferno',vmin = 0,vmax = 1.0)

            plt.scatter(list_x_label1_base,list_y_label1_base,s = 10,c = list_score_label1_base,cmap = 'inferno',marker = '^',vmin = 0,vmax = 1.0)
            plt.xlim(0,255)
            plt.ylim(0,255)
            plt.colorbar()
            plt.imshow(img_convert)
            plt.savefig(new_dir_path_graph_colorbar + str(img_name))
            list_num_point.append(dict_filename_number[img_name]["count"])
            cv2.imwrite(new_dir_path_img  + str(img_name) , img)
            plt.clf()
            
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
            arr_angle = np.array(list_angle)
            arr_score_angle = np.array(list_score_angle)
            degree = -180
            width = 5
            threshold_score = 0.6
            list_degree = []
            while degree <= 180:
                index_angle_1 = np.where(arr_angle >= degree)

                arr_angle_1 = arr_angle[index_angle_1]
                index_angle_2 = np.where(arr_angle_1 < degree + width)
                m = np.sum(arr_score_angle[index_angle_2]) 
                if m >= threshold_score:
                    print(img_name,m,degree)
                    list_degree.append(degree)
                degree = degree + width
            for d in list_degree:
                if d - width in list_degree:
                    list_degree.remove(d - width)
            print(img_name,list_degree)

            
            ax1 = fig.add_subplot(1,1,1)
            ax1.hist(list_angle,bins=72,range=(-180,180),weights = list_score_angle)
            ax1.set_title("distribution of angle",fontsize=12)
    
            ax1.set_xlabel("angle[degree]", size = 10)
            ax1.set_ylabel("score", size = 10)
            plt.tight_layout()
            
            fig.savefig(new_dir_path_graph_angle +  str(img_name))
            plt.clf()
        elif len(list_score_label1) == 0:

            plt.scatter(list_x,list_y,s = 10,c = list_score,cmap ='inferno',vmin = 0,vmax = 1.0)
            #plt.scatter(list_x_label1_1,list_y_label1_1,s = 10,c = "r")        
            plt.xlim(0,255)
            plt.ylim(0,255)
            plt.colorbar()
            plt.imshow(img_convert)
            plt.savefig(new_dir_path_graph_colorbar + str(img_name))
            list_num_point.append(dict_filename_number[img_name]["count"])
            cv2.imwrite(new_dir_path_img  + str(img_name) , img)
            plt.clf()

        # histogram                                                                 

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(list_num_point,range=(0,20),bins = 20) # ,align ='left' )           

    ax.set_title("distribution of detected point",fontsize=12)

    ax.set_xlabel("num of detected point", size = 10)
    ax.set_ylabel("frequency", size = 10)
    plt.xticks(np.arange(0, 20 + 1, 2 ))
    plt.tight_layout()
    fig.savefig(new_dir_path_graph + 'hist_distri.png')
    #print(len(list_num_point))                                                 
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(list_num_point_rm,range=(0,20),bins = 20) # ,align ='left' )        

    ax.set_title("distribution of detected point rm",fontsize=12)

    ax.set_xlabel("num of detected point", size = 10)
    ax.set_ylabel("frequency", size = 10)
    plt.xticks(np.arange(0, 20 + 1, 2 ))
    plt.tight_layout()
    fig.savefig(new_dir_path_graph + 'hist_distri_rm.png')


