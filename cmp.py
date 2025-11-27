import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
sava_dir="cmp_number_400r_20s_20interval/"
if not os.path.exists(sava_dir):
    os.mkdir(sava_dir)
n_recv=400
interval_source=20
recv=np.arange(0,n_recv)

pao=np.arange(0,n_recv,interval_source)
cmp_np=np.zeros([n_recv*int(n_recv/interval_source),3])


for i in recv:
    for j in range(int(n_recv/interval_source)):
        cmp_np[i+j*n_recv,0]=i   #接收点位置
        cmp_np[i + j*n_recv, 1] = j*interval_source   #炮点位置
        cmp_np[i + j*n_recv,2]=(cmp_np[i+j*n_recv,0]+cmp_np[i + j*n_recv, 1])/2
        
sorted_indices = np.argsort(cmp_np[:, 2])
 
# 根据排序后的索引重新排序数组
cmp_np_g = cmp_np[sorted_indices]

all_n=cmp_np_g.shape[0]
cmp_int=[]
for n in tqdm(list(range(all_n))):
    for i in range(n_recv):
        if cmp_np_g[n,2]-i==0:
            # print(cmp_number[n])
            cmp_int.append(cmp_np_g[n])
mp_number=np.array(cmp_int)
df=pd.DataFrame(columns=['i','j','cmp'],data=mp_number)
# print(df)

n_450=[]
# cmp_int_sorted=[]
for i in tqdm(list(range(n_recv))):
    n_450.append((df[df['cmp']==i].shape[0]))     #查看覆盖次数
    df_0=df[df['cmp']==i]
    df_0 = df_0.sort_values(['i','j'])
    if df_0['i'].shape[0] != 0:
        if i==0:
            cmp_int_sorted=df_0.values.reshape(-1,3)
        else:
            temp=df_0.values.reshape(-1,3)
            cmp_int_sorted=np.concatenate((cmp_int_sorted,temp),axis=0)
np.savetxt("temp2.txt",np.array(cmp_int_sorted))

cmp_number = np.loadtxt("temp2.txt")  # 0:  recv  1:shot  2:cmp_distance
# print(cmp_number.shape)
print(cmp_number[:,2])
import pandas as pd

cmp=cmp_number[:,2].astype(int)

df=pd.DataFrame(columns=['i','j','cmp'],data=cmp_number)
for i in range(0,n_recv):
    # n_450.append((df[df['cmp']==i].shape[0]))
    df_0=df[df['cmp']==i]
    # print(df_0)
    df_vi=df_0.values
    np.savetxt(sava_dir+str(i)+"_cmp.txt",df_vi)

        
        
       