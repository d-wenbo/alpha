import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import statistics
import math

def ashmanD(list1,list2):
    myu1 = statistics.mean(list1)
    myu2 = statistics.mean(list2)  
    var1 = sum(( l - myu1)**2 for l in list1) / (len(list1) - 1)
    var2 = sum(( l - myu2)**2 for l in list2) / (len(list2) - 1)
    stdev1 = math.sqrt(var1)
    stdev2 = math.sqrt(var2)
    D = math.sqrt(2) * (abs(myu1 - myu2) / math.sqrt(stdev1**2 + stdev2**2))
    return D 
def searchnum(list,min,max):
    num_sur = 0
    for n in list:
        if min <= n <= max:
            num_sur+=1
    return num_sur





args = sys.argv
num = args[5]
dirname = args[1]
alpha_file = args[2]
other_file = args[3]
not_vertex_file = args[4]
outputname =  dirname + '/result_' + num +'/'

with open(alpha_file,'rb') as f:
    alpha_result = pickle.load(f)

#with open('picklefiles_for_threshold/alpha_old_result' + num +'.pickle','rb') as f:
    #alpha_old_result = pickle.load(f)
with open(other_file,'rb') as f:
    other_result = pickle.load(f)
#with open('picklefiles_for_threshold/other_old_result' + num +'.pickle','rb') as f:
    #other_old_result = pickle.load(f)
with open(not_vertex_file,'rb') as f:
    not_vertex_result = pickle.load(f)
#with open('picklefiles_for_threshold/not_vertex_old_result'+ num +'.pickle','rb') as f:
    #not_vertex_old_result = pickle.load(f)
new_dir_path = outputname
#os.makedirs('AREA07_histfiles_new',exist_ok = True)
os.makedirs(new_dir_path,exist_ok = True)
new_dir_path_graph = new_dir_path 
os.makedirs(new_dir_path_graph,exist_ok = True)

alpha_26 = searchnum(alpha_result,2,6)
other_26 = searchnum(other_result,2,6)
not_vertex_26 = searchnum(not_vertex_result,2,6)
all_26 = alpha_26 + other_26 + not_vertex_26
purity_26 = round(100*(alpha_26/all_26),2)
effi_26 = round(100*(alpha_26/575),1)
print('alpha26;'+str(alpha_26))
print('other26;'+str(other_26))
print('not_vertex26;'+str(not_vertex_26))
print('all_26:' + str(all_26))
print('purity26:'+str(purity_26))
print('effi_26;'+str(effi_26))

alpha_25 = searchnum(alpha_result,2,5)
other_25 = searchnum(other_result,2,5)
not_vertex_25 = searchnum(not_vertex_result,2,5)
all_25 = alpha_25 + other_25 + not_vertex_25
purity_25 = round(100*(alpha_25/all_25),2)
effi_25 = round(100*(alpha_25/575),1)
print('alpha25;'+str(alpha_25))
print('other25;'+str(other_25))
print('not_vertex25;'+str(not_vertex_25))
print('all_25:' + str(all_25))
print('purity25:'+str(purity_25))
print('effi_25;'+str(effi_25))

#print(not_vertex_old_result)

fig = plt.figure()

D1 = ashmanD(other_result,alpha_result)
ax = fig.add_subplot(1,1,1)

ax.hist(other_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='other')
ax.hist(alpha_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='alpha')
ax.legend(loc = 'upper right')
ax.set_xticks(np.linspace(-1,15,17))
ax.set_title('alpha vs other')
ax.set_xlabel('num of detedted')
ax.set_ylabel('frequency')
ax.text(0.8,0.8,'D:' + str(round(D1,3)),transform=ax.transAxes   )
plt.tight_layout()
fig.savefig(new_dir_path_graph + 'alpha_vs_other.png')
plt.clf()

'''
D2 = ashmanD(other_old_result,alpha_old_result)
ax = fig.add_subplot(1,1,1)
ax.hist(other_old_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='other')
ax.hist(alpha_old_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='alpha')
ax.legend(loc = 'upper right')
ax.set_xticks(np.linspace(-1,15,17))
ax.set_title('alpha vs other(old)')
ax.set_xlabel('num of detedted')
ax.set_ylabel('frequency')
ax.text(0.8,0.8,'D:' + str(round(D2,3)),transform=ax.transAxes  )
plt.tight_layout()
fig.savefig(new_dir_path_graph + 'alpha_vs_other_old.png')
plt.clf()

D3 = ashmanD(not_vertex_old_result,alpha_old_result)


plt.hist(not_vertex_old_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='not_vertex')
plt.hist(alpha_old_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='alpha')
plt.legend(loc = 'upper right')
plt.xticks(np.linspace(-1,15,17))
plt.title('alpha vs not vertex(old)')
plt.xlabel('num of detedted')
plt.ylabel('frequency')
plt.text(0.8,0.8,'D:' + str(round(D3,3)),transform=ax.transAxes  )
plt.tight_layout()
fig.savefig(new_dir_path_graph + 'alpha_vs_not_vertex_old.png')
plt.clf()
'''

D4 = ashmanD(not_vertex_result,alpha_result)


plt.hist(not_vertex_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='not_vertex')
plt.hist(alpha_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='alpha')
plt.legend(loc = 'upper right')
plt.xticks(np.linspace(-1,15,17))
plt.title('alpha vs not vertex')
plt.xlabel('num of detedted')
plt.ylabel('frequency')
plt.text(0.8,0.8,'D:' + str(round(D4,3)),transform=ax.transAxes  )
plt.tight_layout()
fig.savefig(new_dir_path_graph + 'alpha_vs_not_vertex.png')
plt.clf()

plt.hist(not_vertex_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='noise')
plt.hist(other_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='beam-interaction')
plt.hist(alpha_result, bins=np.arange(15)+0.5,align='mid',alpha = 0.5,label='alpha')
plt.legend(loc = 'upper right')
plt.xticks(np.linspace(-1,15,17))
#plt.title('alpha vs not vertex')
plt.xlabel('num of detedted')
plt.ylabel('frequency')
#plt.text(0.8,0.8,'D:' + str(round(D4,3)),transform=ax.transAxes  )
plt.tight_layout()
fig.savefig(new_dir_path_graph + 'all.png')



