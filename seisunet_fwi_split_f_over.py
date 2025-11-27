import numpy as np
import math
# print(math.pi)
import matplotlib.pyplot as plt
#

import scipy.ndimage
import time
from numba import jit
import os
from mpi4py import MPI
import torch
from torch.nn.functional import conv1d

from skimage.measure import block_reduce
# import sys
comm = MPI.COMM_WORLD   #Communicator对象包含所有进程
# client_script = 'mpi_fwi.py'
# comm = MPI.COMM_SELF.Spawn(sys.executable, args=[client_script], maxprocs=3)
size = comm.Get_size()
rank = comm.Get_rank()
print('rank',rank,'size ',size,flush=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#Manual seeds for reproducibility
random_seed=30
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import scipy.signal
pi = math.pi

N = 6  # 差分精度

from numpy.fft import fft, ifft
def split_f(data, low_f, high_f):
    nr, nt = data.shape
    data_freq = np.zeros([nr, nt])

    freq = fft(data)
    freq[:, high_f:-high_f] = 0
    freq[:, 0:low_f] = 0
    freq[:, -low_f:] = 0
    # freq[0:1] = 0
    data_freq = ifft(freq)
    return data_freq

"""3Hz"""
# def ricker(dt, favg, Tn):
#     w = np.zeros([Tn])
#     for k in range(Tn):
#         w[k] = 100 * (1 - 2 * ((pi * favg * (k * dt - 1 / favg)) ** 2)) * np.exp(
#             -1.0 * ((pi * favg * (k * dt - 1 / favg)) ** 2))
#     # print(wavelet)
#     return w
"""8Hz"""

def ricker(dt, favg, Tn):
    txz = np.zeros([Tn])
    for k in range(Tn):
        txz[k] = 100 * (1 - 2 * ((pi * favg * (k * dt - 1.1)) ** 2)) * np.exp(
            -1.0 * ((pi * favg * (k * dt - 1.1)) ** 2))

    wavelet = np.zeros([Tn])
    wavelet=txz
    return wavelet


# wavelet_all = np.zeros([Tn])
# wavelet_all[:200] = wavelet
#
# #
# plt.plot(w)
# plt.show()

"""计算差分系数"""
def diff_coef(N):

    a = np.zeros([N + 1])
    for m in range(1, N + 1):
        sx1 = 1.0
        sx2 = 1.0
        for i in range(1, m):
            sx1 = sx1 * (2 * i - 1) * (2 * i - 1)
            sx2 = sx2 * abs((2 * m - 1) * (2 * m - 1) - (2 * i - 1) * (2 * i - 1))
        # print(sx1,sx2)
        for i in range(m + 1, N + 1):
            sx1 = sx1 * (2 * i - 1) * (2 * i - 1)
            sx2 = sx2 * abs((2 * m - 1) * (2 * m - 1) - (2 * i - 1) * (2 * i - 1))
        if m % 2 == 1:
            a[m - 1] = sx1 / (sx2 * (2 * m - 1))
        else:
            a[m - 1] = -sx1 / (sx2 * (2 * m - 1))

        # print("a[{}]={}".format(m - 1, a[m - 1]))
    return a


def ext_model(vp0, NX, NZ, PML):
    V = np.zeros([NX + 2 * PML, NZ + 2 * PML])
    V[PML:PML + NX, PML:PML + NZ] = vp0

    for i in range(PML):
        V[PML:-PML, i] = V[PML:PML + NX, PML]

    for i in range(NZ + PML, NZ + 2 * PML):
        V[PML: - PML, i] = V[PML: PML + NX, NZ + PML - 1]

    for i in range(PML):
        V[i, :] = V[PML, :]
    for i in range(NX + PML, NX + 2 * PML):
        V[i, :] = V[NX + PML - 1, :]
    return V



def pml_ceof(vp,NX_PML,NZ_PML,R,PML,DX,DZ,dt):
    vmax = np.max(vp)
    pmlx = (PML - N) * DX
    pmlz = (PML - N) * DZ
    ddx = np.zeros([NX_PML, NZ_PML])
    ddz = np.zeros([NX_PML, NZ_PML])



    for i in range(NX_PML):
        for j in range(NZ_PML):
            """区域1角点"""
            if i >= 0 and i < PML and j >= 0 and j < PML:
                x = PML - i
                z = PML - j
                ddx[i, j] = -np.log(R) * 2 * vmax * (x ** 2) / (2 * pmlx * (PML ** 2))
                ddz[i, j] = -np.log(R) * 2 * vmax * (z ** 2) / (2 * pmlz * (PML ** 2))

            elif i >= 0 and i < PML and j >= NZ_PML - PML and j < NZ_PML:
                x = PML - i
                z = j - (NZ_PML - PML)
                ddx[i, j] = -np.log(R) * 2 * vmax * (x ** 2) / (2 * pmlx * (PML ** 2))
                ddz[i, j] = -np.log(R) * 2 * vmax * (z ** 2) / (2 * pmlz * (PML ** 2))

            elif i >= NX_PML - PML and i < NX_PML and j >= 0 and j < PML:
                x = i - (NX_PML - PML);
                z = PML - j;
                ddx[i, j] = -np.log(R) * 2 * vmax * (x ** 2) / (2 * pmlx * (PML ** 2))
                ddz[i, j] = -np.log(R) * 2 * vmax * (z ** 2) / (2 * pmlz * (PML ** 2))

            elif i >= NX_PML - PML and i < NX_PML and j >= NZ_PML - PML and j < NZ_PML:
                x = i - (NX_PML - PML)
                z = j - (NZ_PML - PML)
                ddx[i, j] = -np.log(R) * 2 * vmax * (x ** 2) / (2 * pmlx * (PML ** 2))
                ddz[i, j] = -np.log(R) * 2 * vmax * (z ** 2) / (2 * pmlz * (PML ** 2))
            ##上下边界区域##
            elif i >= PML and i < NX_PML - PML and j < PML:
                x = 0
                z = PML - j
                ddx[i, j] = 0
                ddz[i, j] = -np.log(R) * 2 * vmax * (z ** 2) / (2 * pmlz * (PML ** 2))

            elif i >= PML and i < NX_PML - PML and j >= NZ_PML - PML:
                x = 0
                z = j - (NZ_PML - PML)
                ddx[i, j] = 0
                ddz[i, j] = -np.log(R) * 2 * vmax * (z ** 2) / (2 * pmlz * (PML ** 2))
            ##区域2 上下边界区域"##
            elif i < PML and j >= PML and j < NZ_PML - PML:
                x = PML - i
                z = 0
                ddx[i, j] = -np.log(R) * 2 * vmax * (x ** 2) / (2 * pmlx * (PML ** 2))
                ddz[i, j] = 0

            elif i >= NX_PML - PML and j >= PML and j < NZ_PML - PML:
                x = i - (NX_PML - PML)
                z = 0
                ddx[i, j] = -np.log(R) * 2 * vmax * (x ** 2) / (2 * pmlx * (PML ** 2))
                ddz[i, j] = 0

            else:
                ddx[i, j] = 0
                ddz[i, j] = 0

    for i in range(NX_PML):
        for j in range(NZ_PML):
            ddx[i, j] = np.exp(-1.0 * ddx[i, j] * dt)
            ddz[i, j] = np.exp(-1.0 * ddz[i, j] * dt)
    # print(ddx)
    return ddx,ddz

# plt.imshow(ddx)
# plt.colorbar()
# plt.show()
# plt.imshow(ddz)
# plt.colorbar()
# plt.show()



@jit(nopython=True)
def cal(PP_tt,P_now, P_past, P_next, P_x, P_z, pv1, pv2, pt1, pt2, ddx,ddz,vp, a,w,NX_PML,NZ_PML,sx,sz,k,DX,DZ,dt):
    for i in range(N, NX_PML - N):
        for j in range(N, NZ_PML - N):
            # if i==(NX/2) and j== NZ/2:   #set source location
            #     s = 10 * wavelet[k]
            # else:
            #     s=0.0
            px_temp = 0
            pz_temp = 0
            for m in range(0, N):
                """p对x求偏导==P_x"""
                px_temp += a[m] * (P_now[i + m + 1, j] - P_now[i - m, j]) / DX

                """p对z求偏导==P_z"""
                pz_temp += a[m] * (P_now[i, j + m + 1] - P_now[i, j - m]) / DZ

            pt1[i, j] = ddx[i, j] * (pt1[i, j] - px_temp) + px_temp  # 边界上衰减的波场
            pt2[i, j] = ddz[i, j] * (pt2[i, j] - pz_temp) + pz_temp  # 边界上衰减的波场

            P_x[i, j] = px_temp - pt1[i, j]  # 真正的波场
            P_z[i, j] = pz_temp - pt2[i, j]
            # P_x[i, j]=px_temp
            # P_z[i, j] = pz_temp

    for i in range(N, NX_PML - N):
        for j in range(N, NZ_PML - N):
            if i == sx and j == sz:  # set source location
                s = 100 * w[k]
            else:
                s = 0.0

            P_x_x = 0.0
            P_z_z = 0.0
            for m in range(0, N):
                """p_x对x求偏导"""
                P_x_x += a[m] * (P_x[i + m, j] - P_x[i - m - 1, j]) / DX

                """p_z对z求偏导"""
                P_z_z += a[m] * (P_z[i, j + m] - P_z[i, j - m - 1]) / DZ
            #

            pv1[i, j] = ddx[i, j] * (pv1[i, j] - P_x_x) + P_x_x  # 边界上衰减的波场
            pv2[i, j] = ddz[i, j] * (pv2[i, j] - P_z_z) + P_z_z  # 边界上衰减的波场
            PP_tt[i, j] = (vp[i, j] * vp[i, j]) * (P_x_x - pv1[i, j] + P_z_z - pv2[i, j])
            P_next[i, j] = (2 * P_now[i, j] - P_past[i, j]) + dt * dt * (vp[i, j] * vp[i, j]) * (
                        P_x_x - pv1[i, j] + P_z_z - pv2[i, j]) +s
            # P_next[i, j] = (2 * P_now[i, j] - P_past[i, j]) + dt * dt * (vp[i, j] * vp[i, j]) * (P_x_x  + P_z_z ) + s

    P_past = P_now.copy()
    P_now = P_next.copy()

    return PP_tt,P_now, P_past, P_next, P_x, P_z, pv1, pv2, pt1, pt2


@jit(nopython=True)
def cal_inverse(PP_tt,P_now, P_past, P_next, P_x, P_z, pv1, pv2, pt1, pt2, ddx,ddz,vp, a,w,NX_PML,NZ_PML,sx,sz,k,DX,DZ,dt):
    for i in range(N, NX_PML - N):
        for j in range(N, NZ_PML - N):
            # if i==(NX/2) and j== NZ/2:   #set source location
            #     s = 10 * wavelet[k]
            # else:
            #     s=0.0
            px_temp = 0
            pz_temp = 0
            for m in range(0, N):
                """p对x求偏导==P_x"""
                px_temp += a[m] * (P_now[i + m + 1, j] - P_now[i - m, j]) / DX

                """p对z求偏导==P_z"""
                pz_temp += a[m] * (P_now[i, j + m + 1] - P_now[i, j - m]) / DZ

            pt1[i, j] = ddx[i, j] * (pt1[i, j] - px_temp) + px_temp  # 边界上衰减的波场
            pt2[i, j] = ddz[i, j] * (pt2[i, j] - pz_temp) + pz_temp  # 边界上衰减的波场

            P_x[i, j] = px_temp - pt1[i, j]  # 真正的波场
            P_z[i, j] = pz_temp - pt2[i, j]
            # P_x[i, j]=px_temp
            # P_z[i, j] = pz_temp

    for i in range(N, NX_PML - N):
        for j in range(N, NZ_PML - N):
            if i == sx and j == sz:  # set source location
                s = 100 * w[k]
            else:
                s = 0.0

            P_x_x = 0.0
            P_z_z = 0.0
            for m in range(0, N):
                """p_x对x求偏导"""
                P_x_x += a[m] * (P_x[i + m, j] - P_x[i - m - 1, j]) / DX

                """p_z对z求偏导"""
                P_z_z += a[m] * (P_z[i, j + m] - P_z[i, j - m - 1]) / DZ
            #

            pv1[i, j] = ddx[i, j] * (pv1[i, j] - P_x_x) + P_x_x  # 边界上衰减的波场
            pv2[i, j] = ddz[i, j] * (pv2[i, j] - P_z_z) + P_z_z  # 边界上衰减的波场
            PP_tt[i, j] = (vp[i, j] * vp[i, j]) * (P_x_x - pv1[i, j] + P_z_z - pv2[i, j])
            P_next[i, j] = (2 * P_now[i, j] - P_past[i, j]) + dt * dt * (vp[i, j] * vp[i, j]) * (
                        P_x_x - pv1[i, j] + P_z_z - pv2[i, j])
            # P_next[i, j] = (2 * P_now[i, j] - P_past[i, j]) + dt * dt * (vp[i, j] * vp[i, j]) * (P_x_x  + P_z_z ) + s

    P_past = P_now.copy()
    P_now = P_next.copy()

    return PP_tt,P_now, P_past, P_next, P_x, P_z, pv1, pv2, pt1, pt2



# if rank == 0:


DX = 20
DZ = 20
dt = 0.002
NX = 175
NZ = 450

PML = 30

NR = 450
NS = 45
NX_PML = NX + 2 * PML
NZ_PML = NZ + 2 * PML

delta_s = 10
delta_r = 1
bound = 23

NX_PML = NX + 2 * PML
NZ_PML = NZ + 2 * PML

favg = 10
Tn = 3000  # 时间长度
batch_size=256

delta_1=20
delta_2=40
noise_para=0.05

SNR=0

n_cmp = 3

n_channel=3

import torch
from torch import nn
import numpy as np
import time
import scipy.ndimage
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import os

from ParamConfig import *
from PathConfig import *
from LibConfig import *

################################################
########             NETWORK            ########
################################################
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
random_seed=30
torch.manual_seed(random_seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Here indicating the GPU you want to use. if you don't have GPU, just leave it.
cuda_available = torch.cuda.is_available()
device   = torch.device("cuda" if cuda_available else "cpu")
# device   = torch.device("cuda")
# net = UnetModel(n_classes=Nclasses,in_channels=Inchannels,is_deconv=True,is_batchnorm=True)
# if torch.cuda.is_available():
#     net.cuda()

# Optimizer we want to use
# optimizer = torch.optim.Adam(net.parameters(),lr=LearnRate)

# If ReUse, it will load saved model from premodelfilepath and continue to train



# if rank ==0:
#     bochangfile = 'bochang_new/'
#     if not os.path.exists(bochangfile):
#         os.mkdir(bochangfile)

#     recordfile = 'mar2_45_450_f_6_dx_15_no_lowf_SNR_5/'
#     if not os.path.exists(recordfile):
#         os.mkdir(recordfile)

#     resultfile = 'fwi_delta_cmp_f_6_20230129_lienar_gen_no_lowf_SNR_5_easy/'
#     if not os.path.exists(resultfile):
#         os.mkdir(resultfile)

# recordfile = 'mar2_45_450_f_6_dx_15_no_lowf_SNR_5/'
# resultfile = 'fwi_delta_cmp_f_6_20230129_lienar_gen_no_lowf_SNR_5_easy/'
lowcut_f=2

if rank ==0:
    bochangfile = 'bochang_new/'
    if not os.path.exists(bochangfile):
        os.mkdir(bochangfile)

#     recordfile='over_{}_{}_f_{}_dx_{}_snr_{}_lowcut{}Hz/'.format(NS,NR,favg,DX,str(SNR),lowcut_f)
#     if not os.path.exists(recordfile):
#         os.mkdir(recordfile)

    resultfile = 'cmpfwi_over_adam_f_{}_linear_model_SNR_{}_lowcut{}Hz/'.format(favg,str(SNR),lowcut_f)
    if not os.path.exists(resultfile):
        os.mkdir(resultfile)
        


recordfile='over_{}_{}_f_{}_dx_{}_snr_{}_lowcut{}Hz/'.format(NS,NR,favg,DX,str(SNR),lowcut_f)
resultfile = 'cmpfwi_over_adam_f_{}_linear_model_SNR_{}_lowcut{}Hz/'.format(favg,str(SNR),lowcut_f)


w = ricker(dt, favg, Tn)
w=w.astype(np.float32)
"""3hz高通滤波"""
# wavelet_long=np.zeros([5000])
# wavelet_long[:Tn]=w[:]
# low_freq=int(3/(1/dt/5000))

# fp=fft(wavelet_long)

# fp[ :low_freq] = 0
# fp[ -low_freq:] = 0
# for i in range(8):
#     fp[low_freq+i]=0.125*i*fp[low_freq+i]
#     fp[-low_freq-i]=0.125*i*fp[-low_freq-i]
# import scipy.ndimage
# fp=fp.astype(np.float32)
# fp=scipy.ndimage.filters.gaussian_filter(fp, sigma=5)



# wavelet_long=ifft(fp)
# wavelet_long=wavelet_long.astype(np.float32)

# w[:]=wavelet_long[:Tn]

# vp = np.ones([NX, NZ]) * 3000
# vp[150:, :] = 5000
# den = np.ones([NX, NZ]) * 3000
# den[150:, :] = 4000

import torch
from torch import nn
import numpy as np
import time
import scipy.ndimage

import matplotlib.pyplot as plt




vp_real = np.fromfile("overthrust_450_175.bin", dtype=np.float32).reshape([NZ, NX])
vp = np.fromfile("overthrust_linear_1500_3800.bin", dtype=np.float32).reshape([NZ, NX])

# vp = np.fromfile("cmp_over_result_f_10_snr0lowcut2Hz_nolog.bin", dtype=np.float32).reshape([NZ, NX])
vp[:,:bound]=vp_real[:,:bound]
vp = vp.T

model_initial=np.fromfile("overthrust_linear_1500_3800.bin", dtype=np.float32).reshape([NZ, NX])


vp_prior = np.fromfile( 'over_prior.bin', dtype=np.float32).reshape([NZ, NX])
vp_prior=vp_prior.T



vp = ext_model(vp, NX, NZ, PML)
print(vp.shape)
# vp=vp.reshape([1,-1])
# vp=np.squeeze(vp)
# den = ext_model(den, NX, NZ, PML)
obj_f_list = []
sx_list = [int(np.ceil(PML + 0)) for i in range(0, NS * delta_s, delta_s)]
sz_list = [int(np.ceil(PML + i)) for i in range(0, NS * delta_s, delta_s)]  # 震源深度  每一炮

rx_list = [int(np.ceil(PML + 0)) for i in range(0, NR * delta_r, delta_r)]
rz_list = [int(np.ceil(PML + i)) for i in range(0, NR * delta_r, delta_r)]  # 接收点深度  每个点都接收

a = diff_coef(N)

""""分频"""
N_filter = 3
fenpin =1  # 是否分频

f_list = [i for i in range(lowcut_f+2, int(2 * favg+1), 2)]

iter_f_list = [50 * i for i in range(len(f_list) + 1)]
comm.Barrier()
#反演最高到3Hz 测井做了同样的滤波
for i in range(1,len(f_list)):
    f_list[i]=lowcut_f+2


p1_adam=0.9
p2_adam=0.999
theta_adam=1e-8
s_adam=np.zeros([NX_PML, NZ_PML])
r_adam=np.zeros([NX_PML, NZ_PML])
s_adam_1=np.zeros([NX_PML, NZ_PML])
r_adam_1=np.zeros([NX_PML, NZ_PML])

lt = 10
sm=np.zeros([lt, (NX-bound)* NZ])
ym=np.zeros([lt, (NX-bound)*NZ])

p=np.zeros([(NX-bound)*NZ])

"""带通滤波"""
# low_freq=int(6/(1/dt/Tn))
# high_freq = int(50/(1/dt/Tn))
#
# b = split_f(w.reshape([1, -1]), low_f=low_freq, high_f=high_freq)
# w=b[0].astype(np.float32)

from nnmodels import inverse_model,UnetModel,UnetModelDropout,Generator,Seis_UnetModelDropout
import argparse
import os
# 注意：更改网络结构时此处需要修改
from torch.utils import data
from torch import nn, optim
from datetime import datetime
from tqdm import tqdm
import hashlib
import torch

if rank == 0:

    inverse_net = Seis_UnetModelDropout(n_classes=1, in_channels=n_cmp * n_channel, NX=NX, is_deconv=True,
                                        is_batchnorm=True)
    gen_net = Generator()
    if torch.cuda.is_available():
        inverse_net.cuda()
        gen_net.cuda()
    gen_net.train()
    inverse_net.train()
    criterion = nn.MSELoss()

    # criterion = nn.SmoothL1Loss()
    #
    # optimizer = inverse_net.optimizer

    optimizer = torch.optim.Adam([
        {'params': inverse_net.parameters(), 'lr': 0.003,'weight_decay ':1e-4}
    ])
    optimizer_gen = torch.optim.Adam([
        {'params': gen_net.parameters(), 'lr': 0.003,'weight_decay ':1e-4}
    ])

comm.Barrier()

"""定义速度和密度模型"""


R = 1e-10
iter_i=0

if rank==0:
    obj_f = 0
    obj_f_list = []
obj_f = comm.bcast(obj_f if rank == 0 else None, root=0)

comm.Barrier()
# print(11111,rank)
vp1= np.zeros([NX_PML, NZ_PML],dtype=np.float32)
vp2= np.zeros([NX_PML, NZ_PML],dtype=np.float32)
delta_grad= np.zeros([NX_PML, NZ_PML],dtype=np.float32)

label_dsp_dim = [1,NX]


vp_bg=vp.copy()
totalrank=size

weight = np.loadtxt("weight_marmousi.txt")
weight = weight / weight.max()

p = np.mean(weight) + np.std(weight)

p = torch.tensor(p).float()
print(np.mean(weight))
print(np.std(weight))
weight = torch.tensor(weight).float()
wave = ricker(dt, favg, Tn)


from bruges.filters import wavelets
wavelet_noise = wavelets.ricker(0.8, dt, favg)
wavelet_noise=wavelet_noise/wavelet_noise.max()
wavelet_noise = torch.tensor(wavelet_noise).unsqueeze(dim=0).unsqueeze(dim=0)

def add_noise_torch(data_raw,wavelet_nosie,snr_torch,Tn):
    data_raw_temp=data_raw.reshape([-1,1,Tn])
    random_noise=torch.randn_like(data_raw_temp)
    
    tmp_noise=[]
    for i in range(random_noise.size()[0]):
        tmp_noise.append(conv1d(random_noise[[i],:], wavelet_noise, padding=int(wavelet_noise.shape[-1] / 2))) 
    synth=torch.cat(tmp_noise,dim=0)

    snr_torch=torch.tensor(SNR / 10)
    ps = torch.mean(torch.pow(data_raw_temp, 2))
    pn1 = torch.mean(torch.pow(synth, 2))
    k = torch.sqrt(ps / (torch.pow(10,snr_torch )))  #
    noise_data_xianggan = synth * k
    record_xianggan_noise = data_raw_temp + noise_data_xianggan[...,:data_raw_temp.size()[-1]]
    
    record_xianggan_noise=record_xianggan_noise.reshape(data_raw.shape)
    return record_xianggan_noise

if rank == 0:
    import numpy as np
    well_loc = [300,300]


    all_record_0 = np.zeros([NS, NR, Tn])
    for i in range(NS):
        # data_s = np.fromfile(record_dir + str(i) + '.bin', dtype=np.float32).reshape([450, 4096])
        try:
            data_s = np.fromfile(recordfile + str(i) + '.bin', dtype=np.float32).reshape([NR, Tn])
        except:
            data_s = np.fromfile(recordfile + str(i) + '.dat', dtype=np.float32).reshape([NR, Tn])



        all_record_0[i, :, :] = data_s[:, :]


    n_cmp = n_cmp
    cmp_8_data = np.zeros([NR, n_cmp, Tn + NS + NR])  # 中心点覆盖次数为8的cmp道集

    import os

    # cmp_dir = "cmp_number_mar160/"
    cmp_dir = "cmp_number/"

    if not os.path.exists(cmp_dir):
        os.mkdir(cmp_dir)



    n_450 = []
    for i in range(0, NR):

        ar = np.loadtxt(cmp_dir + str(i) + "_cmp.txt")
        ar = ar.reshape([-1, 3])
        # print(ar)
        if ar.shape[0] != 0:
            # print(i)

            cmp_n = np.zeros([ar.shape[0], Tn + NS + NR])  # 求单个中心点的道集
            # print(ar.shape[0])
            for n in range(ar.shape[0]):

                # print(df_0)
                n_recv = ar[n, 0]
                n_shot = ar[n, 1] 
                n_recv = ar[n, 0]
                n_shot = ar[n, 1]
                # print(n_shot,n_recv)
                weight_d = np.arange(0, NS)
                weight_d = weight_d.astype(np.float32)
                train_indecies = [int(n_shot / delta_s)]
                train_indecies = np.array(train_indecies)
                source_coding = np.zeros(NS)

                for ii in range(weight_d.shape[0]):

                    wd = train_indecies - weight_d[ii]
                    wd = np.abs(wd)
                    wd = wd
                    # print(ww)
                    for mm in range(wd.shape[0]):
                        if wd[mm] != 0:
                            wd[mm] = 1 / wd[mm]
#                                 wd[mm] = 1 

                        else:
                            wd[mm] = 1
                    source_coding[ii] = np.sum(wd)

                source_coding = source_coding / source_coding.max()

                weight_d = np.arange(0, NR)
                weight_d = weight_d.astype(np.float32)
                train_indecies = [n_recv]
                train_indecies = np.array(train_indecies)
                receiv_coding = np.zeros(NR)

                for ii in range(weight_d.shape[0]):
                    wd = train_indecies - weight_d[ii]
                    wd = np.abs(wd)
                    wd = wd
                    # print(ww)
                    for mm in range(wd.shape[0]):
                        if wd[mm] != 0:
                            wd[mm] = 1 / wd[mm]
#                                 wd[mm] = 1 

                        else:
                            wd[mm] = 1
                    receiv_coding[ii] = np.sum(wd)

                receiv_coding = receiv_coding / receiv_coding.max()


                # print(n_shot,n_recv)

                if n < int(ar.shape[0] - 1) / 2:

                    cmp_n[2 * n + 1, :Tn] = all_record_0[int(n_shot / delta_s), int(n_recv), :]
                    cmp_n[2 * n + 1, Tn:Tn + NS] = source_coding[:]
                    cmp_n[2 * n + 1, Tn + NS:Tn + NS + NR] = receiv_coding[:]


                else:


                    cmp_n[((ar.shape[0] - n - 1) * 2), :Tn] = all_record_0[int(n_shot / delta_s), int(n_recv), :]
                    cmp_n[((ar.shape[0] - n - 1) * 2), Tn:Tn + NS] = source_coding[:]
                    cmp_n[((ar.shape[0] - n - 1) * 2), Tn + NS:Tn + NS + NR] = receiv_coding[:]


                # cmp_n[n]=all_record[int(n_shot/10),int(n_recv)] #求单个中心点的道集

                # cmp_stack[i]+=all_record[int(n_shot/10),int(n_recv)]  #求叠加道集
                # cmp_stack[i]=cmp_stack[i]/ar.shape[0]     #求叠加道集

            # cmp_n.astype(np.float32).tofile('cmp_'+str(i)+'_stack_'+str(n)+'_4096.bin') #求单个中心点的道集
            if ar.shape[0] > n_cmp:
                cmp_8_data[i, :, :] = cmp_n[-n_cmp:, :]
            else:
                cmp_8_data[i, -ar.shape[0]:, :] = cmp_n[:, :]
                for k in range(n_cmp - ar.shape[0]):
                    cmp_8_data[i, k, :] = cmp_n[0, :]

            # cmp_stack.astype(np.float32).tofile('cmp_stack_450_4096.bin')

#             print(cmp_8_data.shape)
#             t_start=1
#             for n_trace in range(NR):
#                 for tk in range(1,Tn):
#                     if cmp_8_data[n_trace,-1,tk]-cmp_8_data[n_trace,-1,tk-1]>=0.01:
#                         t_start=tk
#                         break
#                     break
#             #     print(t_start)
#                 for i_cmp in range(n_cmp):
#                     cmp_8_data[n_trace,n_cmp-i_cmp-1,0:t_start+455+85*i_cmp]=0


#             """test: to emplify the amplitude of the weak signal"""
#             for i in range(seismic_data_raw.shape[0]):
#                 for j in range(seismic_data_raw.shape[1]):
#                     for k in range(seismic_data_raw.shape[2]):
#                         if seismic_data_raw[i,j,k] >0:
#                             seismic_data_raw[i, j, k]=1
#                         else:
#                             seismic_data_raw[i, j, k] = -1

    seismic_data_raw = cmp_8_data.copy()

    # print(seismic_data_raw.shape)

    seismic_data = np.zeros([NZ, n_channel, n_cmp, NS + NR + Tn])

    for i in range(n_channel):
        seismic_data[:, i, :, :] = seismic_data_raw[:, :, :]

    for x_i in range(NZ):
        if x_i ==0:
            for jj in range(n_channel-2):
                seismic_data[x_i, jj, :, :] = seismic_data_raw[x_i, :, :]
            seismic_data[x_i, n_channel-2, :, :] = seismic_data_raw[x_i+1, :, :]
            seismic_data[x_i, n_channel-1, :, :] = seismic_data_raw[x_i+2, :, :]
        if x_i==1:
            seismic_data[x_i, 0, :, :] = seismic_data_raw[x_i-1, :, :]
            seismic_data[x_i, 1, :, :] = seismic_data_raw[x_i-1, :, :]
            for jj in range(2,n_channel-2):
                seismic_data[x_i, jj, :, :] = seismic_data_raw[x_i, :, :]

            seismic_data[x_i, n_channel-2, :, :] = seismic_data_raw[x_i+1, :, :]
            seismic_data[x_i, n_channel-1, :, :] = seismic_data_raw[x_i+2, :, :]
        if x_i >=2 and x_i <=(NZ-3):
            seismic_data[x_i, 0, :, :] = seismic_data_raw[x_i-2, :, :]
            seismic_data[x_i, 1, :, :] = seismic_data_raw[x_i-1, :, :]

            for jj in range(2,n_channel-2):
                seismic_data[x_i, jj, :, :] = seismic_data_raw[x_i, :, :]

            seismic_data[x_i, n_channel-2, :, :] = seismic_data_raw[x_i+1, :, :]
            seismic_data[x_i, n_channel-1, :, :] = seismic_data_raw[x_i+2, :, :]
        if x_i==(NZ-2):
            seismic_data[x_i, 0, :, :] = seismic_data_raw[x_i-2, :, :]
            seismic_data[x_i, 1, :, :] = seismic_data_raw[x_i-1, :, :]
            for jj in range(2,n_channel-2):
                seismic_data[x_i, jj, :, :] = seismic_data_raw[x_i, :, :]
            seismic_data[x_i, n_channel-2, :, :] = seismic_data_raw[x_i+1, :, :]
            seismic_data[x_i, n_channel-1, :, :] = seismic_data_raw[x_i+1, :, :]
        if x_i == (NZ-1):
            seismic_data[x_i, 0, :, :] = seismic_data_raw[x_i-2, :, :]
            seismic_data[x_i, 1, :, :] = seismic_data_raw[x_i-1, :, :]
            for jj in range(2,n_channel):
                seismic_data[x_i, jj, :, :] = seismic_data_raw[x_i, :, :]
                seis_mean = np.max(seismic_data[:, :, :, :Tn], axis=(0, -1), keepdims=True)
    seis_std = np.min(seismic_data[:, :, :, :Tn], axis=(0, -1), keepdims=True)
    seismic_data[:, :, :, :Tn] = (seismic_data[:, :, :, :Tn] - seis_std) / (seis_mean - seis_std)
    seismic_data = seismic_data.reshape(NR, n_channel * n_cmp, Tn + NR + NS)
    # print(seismic_data.shape)

    seismic_data_temp = np.zeros([seismic_data.shape[0], seismic_data.shape[1], NX * 10]) #  设置增强地震道的长度  nx*10
    seismic_data_temp[:, :, -(NR + NS):] = seismic_data[:, :, -(NR + NS):]
    seismic_data_temp[:, :, :NX * 10 - (NR + NS)] = seismic_data[:, :, int((pi * 5 * 1.1) ** 2):int((pi * 5 * 1.1) ** 2)+NX * 10 - (NR + NS)] 
    seismic_data = seismic_data_temp.copy()


    # print(seismic_data.shape)

    unlabeled_seismic_data=seismic_data.copy()
#             unlabeled_seismic_data = np.delete(seismic_data, well_loc, axis=0)

    labeled_seismic_data = seismic_data[well_loc]

    unlabeled_seismic_data = torch.tensor(unlabeled_seismic_data).float()
    labeled_seismic_data = torch.tensor(labeled_seismic_data).float()
    seismic_data = torch.tensor(seismic_data).float()
    if torch.cuda.is_available():
        seismic_data=seismic_data.cuda()
        unlabeled_seismic_data=unlabeled_seismic_data.cuda()
        labeled_seismic_data=labeled_seismic_data.cuda()



comm.Barrier()



for iter_i in range(40):
    record_all = np.zeros([len(sx_list), NR,Tn])
    comm.Barrier()
    print(iter_i,flush=True)
    ddx, ddz = pml_ceof(vp, NX_PML, NZ_PML, R, PML, DX, DZ, dt)
    g = np.zeros([NX_PML, NZ_PML])
    g_ = np.zeros([NX_PML, NZ_PML])
    obj_f = 0
    for si in range(int(np.ceil((len(sx_list)+1)/totalrank))):
        # print(si,flush = True)
        s_sum = rank+si*totalrank
        if s_sum<len(sx_list):
            # print('rank: ', rank, 'and s_sum: ', s_sum, sx_list[s_sum], sz_list[s_sum], flush=True)
            # while (s_sum <3):
            # for s_num in range(3):
            #     if rank==( s_sum % 3 +1):

            ddx, ddz = pml_ceof(vp, NX_PML, NZ_PML, R, PML, DX, DZ, dt)
            import time
            # print(s_sum, rank,333333333333)
            record = np.zeros([NR, Tn])  ##地震记录
            """初始化波场参数"""
            P_now = np.zeros([NX_PML, NZ_PML])
            P_past = np.zeros([NX_PML, NZ_PML])
            P_next = np.zeros([NX_PML, NZ_PML])
            pt1 = np.zeros([NX_PML, NZ_PML])
            pt2 = np.zeros([NX_PML, NZ_PML])
            pv1 = np.zeros([NX_PML, NZ_PML])
            pv2 = np.zeros([NX_PML, NZ_PML])
            P_x = np.zeros([NX_PML, NZ_PML])  # vx
            P_z = np.zeros([NX_PML, NZ_PML])  # vz
            PP_tt = np.zeros([NX_PML, NZ_PML])
            bochang = np.zeros([Tn, NX_PML, NZ_PML])

            """对子波滤波"""
            if fenpin == 1:

                for i_for_f in range(1, len(iter_f_list)):

                    if iter_i - iter_f_list[i_for_f - 1] >= 0 and iter_i - iter_f_list[i_for_f] < 0:
                        print(f_list[i_for_f - 1], 'f_  ', flush=True)
                        aa, bb = scipy.signal.butter(N_filter, 2 * f_list[i_for_f - 1] / (1 / dt), btype='lowpass',
                                                     analog=False, output='ba',
                                                     fs=None)  # low_freq/ Nyquist freq == low_freq/（dt/2）
                        wave = ricker(dt, favg, Tn)
                        wave = wave.astype(np.float32)
                        #                 wave_temp=np.zeros([Tn])
                        #                 wave_temp[1000:]=wave[:-1000]
                        #                 wave_temp = scipy.signal.filtfilt(aa, bb, wave_temp)
                        #                 wave[:-1000] = wave_temp[1000:]
                        wave = scipy.signal.filtfilt(aa, bb, wave)
                        wave = wave.astype(np.float32)

            # print(vp.shape,'vp\n',a.shape,'a\n',w.shape,'w\n',ddx.shape,'ddx\n',ddz.shape,'ddz\n',len(sx_list),'sx_list\n')
            for k in range(Tn):
                # print(k,flush = True)
                PP_tt, P_now, P_past, P_next, P_x, P_z, pv1, pv2, pt1, pt2 = cal(PP_tt, P_now, P_past, P_next, P_x, P_z,
                                                                                 pv1, pv2, pt1, pt2, ddx,
                                                                                 ddz, vp, a, wave, NX_PML, NZ_PML,
                                                                                 sx_list[s_sum], sz_list[s_sum], k,DX,DZ,dt)

                for i in range(NR):
                    record[i, k] = P_now[rx_list[i], rz_list[i]]

                # PP_tt.tofile(bochangfile + str(k) + '.dat')
                bochang[k] = PP_tt
            # print('rank: ', rank, 'cal wavefield', flush=True)
            record_all[s_sum,:,:]=record[:,:]
            try:
                record_raw = np.fromfile(recordfile + str(s_sum) + '.dat', dtype=np.float32).reshape([NR, Tn])
            except:
                record_raw = np.fromfile(recordfile + str(s_sum) + '.bin', dtype=np.float32).reshape([NR, Tn])
            if fenpin == 1:
                for i_for_f in range(1, len(iter_f_list)):
                    if iter_i - iter_f_list[i_for_f - 1] >= 0 and iter_i - iter_f_list[i_for_f] < 0:
                        aa, bb = scipy.signal.butter(N_filter, 2 * f_list[i_for_f - 1] / (1 / dt), btype='lowpass',
                                                     analog=False,
                                                     output='ba', fs=None)  # low_freq/ Nyquist freq == low_freq/（dt/2）
                        record = scipy.signal.filtfilt(aa, bb, record)
                        record_raw = scipy.signal.filtfilt(aa, bb, record_raw)


            # f_dt = 2e-3
            # f_nt = Tn
#             high_freq = int(3 / (1 / f_dt / f_nt))
#             fp = fft(record_raw)
#             fp[:, high_freq:-high_freq] = 0
#             record_raw = ifft(fp)
#             record_raw=record_raw.astype(np.float32)
            
#             fp = fft(record)
#             fp[:, high_freq:-high_freq] = 0
#             record = ifft(fp)
#             record=record.astype(np.float32)
            
            
            res = record - record_raw
            res = res.astype(np.float32)

            res=np.nan_to_num(res)

            obj_f += np.sum(res ** 2)
            print(rank,obj_f,flush=True)
            Pi_now = np.zeros([NX_PML, NZ_PML])
            Pi_past = np.zeros([NX_PML, NZ_PML])
            Pi_next = np.zeros([NX_PML, NZ_PML])
            pti1 = np.zeros([NX_PML, NZ_PML])
            pti2 = np.zeros([NX_PML, NZ_PML])
            pvi1 = np.zeros([NX_PML, NZ_PML])
            pvi2 = np.zeros([NX_PML, NZ_PML])
            Pi_x = np.zeros([NX_PML, NZ_PML])  # vx
            Pi_z = np.zeros([NX_PML, NZ_PML])  # vz
            PPi_tt = np.zeros([NX_PML, NZ_PML])
            # """反传波场"""
            for n in list(range(Tn))[::-1]:
                # print(n, flush = True)
                for i in range(NR):
                    Pi_now[rx_list[i], rz_list[i]] += res[i, n]

                PPi_tt, Pi_now, Pi_past, Pi_next, Pi_x, Pi_z, pvi1, pvi2, pti1, pti2 = cal_inverse(PPi_tt, Pi_now, Pi_past, Pi_next,Pi_x, Pi_z, pvi1, pvi2, pti1, pti2, ddx,ddz, vp, a, wave, NX_PML, NZ_PML,  sx_list[s_sum],sz_list[s_sum], n,DX,DZ,dt)

                # PP_tt_n=np.fromfile(bochangfile+str(n)+'.dat').reshape([NX_PML, NZ_PML])
                PP_tt_n = bochang[n]
                # if n%50==0:
                #     plt.imshow(PP_tt_n)
                #     plt.pause(0.5)
                #     plt.clf()

                g += (-2) * PP_tt_n * Pi_now / (vp * vp * vp)

                # g=np.nan_to_num(g)


                # max_pptt=np.max(np.abs(PP_tt_n))
                # print(max_pptt)
                g_ += ((-2) * PP_tt_n / (vp * vp * vp)) * ((-2) * PP_tt_n / (vp * vp * vp))


    comm.Barrier()
    g = comm.reduce(g, root=0, op=MPI.SUM)
    # # g_sum[PML:-PML, PML:-PML].T.astype(np.float32).tofile(str(iter_i) + 'grad.dat')
    g_ = comm.reduce(g_, root=0, op=MPI.SUM)
    obj_f = comm.reduce(obj_f, root=0, op=MPI.SUM)
    # obj_f_0=obj_f
    # comm.Bcast(obj_f, root=0)
    print("end",rank,flush=True)
    # print(rank, flush=True)
    # print(s_sum, rank, 2222222222222)
    # comm.Barrier()


    if rank==0:
    # print(1111,flush=True)
        # 网络中的输入是 ns nt nr 而record_all 是NS NR NT

        g_max=np.max(np.abs(g_[PML + bound:-PML, PML:-PML]))
#         print(g_max)

        g_ = g_+g_max/1000
#         if iter_i >150:
#             grad_T = g / g_
        if iter_i < 80:
            grad_T = g
            delta_grad = np.zeros([NX_PML, NZ_PML])
            grad_min, max_grad = np.percentile(abs(grad_T[PML + bound:-PML, PML:-PML]), [2, 98])
            delta_grad[PML + bound:-PML, PML:-PML] = grad_T[PML + bound:-PML, PML:-PML]/max_grad
            delta_grad=delta_grad.astype(np.float32)
        if iter_i >= 80:
            grad_T = g/ g_
            delta_grad = np.zeros([NX_PML, NZ_PML])
            grad_min, max_grad = np.percentile(abs(grad_T[PML + bound:-PML, PML:-PML]), [2, 98])
            delta_grad[PML + bound:-PML, PML:-PML] = grad_T[PML + bound:-PML, PML:-PML]/max_grad
            delta_grad=delta_grad.astype(np.float32)
            
#         grad_T = g/ g_
#         delta_grad = np.zeros([NX_PML, NZ_PML])
#         grad_min, max_grad = np.percentile(abs(grad_T[PML + bound:-PML, PML:-PML]), [2, 98])
#         delta_grad[PML + bound:-PML, PML:-PML] = grad_T[PML + bound:-PML, PML:-PML]/max_grad
#         delta_grad=delta_grad.astype(np.float32)     

#         grad_T = g/ g_
#         delta_grad = np.zeros([NX_PML, NZ_PML])
#         grad_min, max_grad = np.percentile(abs(grad_T[PML + bound:-PML, PML:-PML]), [2, 98])
#         delta_grad[PML + bound:-PML, PML:-PML] = grad_T[PML + bound:-PML, PML:-PML]/max_grad
#         delta_grad=delta_grad.astype(np.float32)
#         grad_T = g/ g_
#         delta_grad = np.zeros([NX_PML, NZ_PML])
#         grad_min, max_grad = np.percentile(abs(grad_T[PML + bound:-PML, PML:-PML]), [2, 98])
#         delta_grad[PML + bound:-PML, PML:-PML] = grad_T[PML + bound:-PML, PML:-PML]/max_grad
#         delta_grad=delta_grad.astype(np.float32)
        
        # if iter_i==350:
        #     s_adam=np.zeros([NX_PML, NZ_PML])
        #     r_adam=np.zeros([NX_PML, NZ_PML])
        #     s_adam_1=np.zeros([NX_PML, NZ_PML])
        #     r_adam_1=np.zeros([NX_PML, NZ_PML])
        #####adam
        
#         if  iter_i >150 and iter_i<300:
#         grad_T = g
#         delta_grad = np.zeros([NX_PML, NZ_PML])
#         grad_min, max_grad = np.percentile(abs(grad_T[PML + bound:-PML, PML:-PML]), [2, 98])
#         delta_grad[PML + bound:-PML, PML:-PML] = grad_T[PML + bound:-PML, PML:-PML]/max_grad
#         delta_grad=delta_grad.astype(np.float32)

#         s_adam[PML + bound:-PML, PML:-PML] = p1_adam * s_adam[PML + bound:-PML, PML:-PML] + (1 - p1_adam) * delta_grad[PML + bound:-PML,PML:-PML]
#         r_adam[PML + bound:-PML, PML:-PML] = p2_adam * r_adam[PML + bound:-PML, PML:-PML] + (1 - p2_adam) * delta_grad[PML + bound:-PML,PML:-PML] * delta_grad[PML + bound:-PML,PML:-PML]
#         s_adam_1[PML + bound:-PML, PML:-PML] = s_adam[PML + bound:-PML, PML:-PML] / (1 - np.power(p1_adam, iter_i + 1))
#         r_adam_1[PML + bound:-PML, PML:-PML] = r_adam[PML + bound:-PML, PML:-PML] / (1 - np.power(p2_adam, iter_i + 1))
#         delta_grad[PML + bound:-PML, PML:-PML] = s_adam_1[PML + bound:-PML, PML:-PML] / (np.sqrt(r_adam_1[PML + bound:-PML, PML:-PML]) + theta_adam)
#         if iter_i < 20:
#             for nx_i in range(NX_PML):
#                 for nz_i in range(NZ_PML):
#                     delta_grad[nx_i,nz_i]=delta_grad[nx_i,nz_i]*1/(1+np.power((nx_i/(PML+100)),16))
#                     s_adam[nx_i,nz_i]=s_adam[nx_i,nz_i]*1/(1+np.power((nx_i/(PML+100)),16))
#                     r_adam[nx_i,nz_i]=r_adam[nx_i,nz_i]*1/(1+np.power((nx_i/(PML+100)),16))
#                     s_adam_1[nx_i,nz_i]=s_adam_1[nx_i,nz_i]*1/(1+np.power((nx_i/(PML+100)),16))
#                     r_adam_1[nx_i,nz_i]=r_adam_1[nx_i,nz_i]*1/(1+np.power((nx_i/(PML+100)),16))
                    
#         if iter_i >=20 and iter_i <50:
#             for nx_i in range(NX_PML):
#                 for nz_i in range(NZ_PML):
#                     delta_grad[nx_i, nz_i] = delta_grad[nx_i,nz_i]*1 / (1 + np.power((nx_i / (PML + 100+75*(iter_i-20)/(50-20))), 16))
#                     s_adam[nx_i,nz_i]=s_adam[nx_i,nz_i]*1/(1 + np.power((nx_i / (PML + 100+75*(iter_i-20)/(50-20))), 16))
#                     r_adam[nx_i,nz_i]=r_adam[nx_i,nz_i]*1/(1 + np.power((nx_i / (PML + 100+75*(iter_i-20)/(50-20))), 16))
#                     s_adam_1[nx_i,nz_i]=s_adam_1[nx_i,nz_i]*1/(1 + np.power((nx_i / (PML + 100+75*(iter_i-20)/(50-20))), 16))
#                     r_adam_1[nx_i,nz_i]=r_adam_1[nx_i,nz_i]*1/(1 + np.power((nx_i / (PML + 100+75*(iter_i-20)/(50-20))), 16))
#         import scipy.ndimage  ##lbfgs
#         # delta_grad[PML + bound+5:-PML-5, PML:-PML] = scipy.ndimage.filters.gaussian_filter(delta_grad[PML + bound+5:-PML-5, PML:-PML], sigma=3)
#         print('delta_grad', delta_grad, flush=True)


#         if iter_i==0:
#             s0=0.00006*delta_grad[PML + bound:-PML, PML:-PML]
#             y0=0.00001*delta_grad[PML + bound:-PML, PML:-PML]

#             # pass
#         else:
#             s0 = vp[PML + bound:-PML, PML:-PML]-vp_pre[PML + bound:-PML, PML:-PML]
#             y0 = delta_grad[PML + bound:-PML, PML:-PML]-grad_pre[PML + bound:-PML, PML:-PML]
#         s0=s0.reshape([(NX-bound)*NZ])
#         y0=y0.reshape([(NX-bound)*NZ])
# #         hdiag=np.dot(s0,y0)/np.dot(y0,y0)
#         hdiag=1
#         print('hdiag', hdiag, flush=True)
#         if iter_i < lt :
#           # if iter_i==0:
#           #     pass
#           # else:

#             sm[iter_i]=s0
#             ym[iter_i] = y0

#             q = np.zeros([iter_i + 1, (NX-bound)*NZ])
#             r = np.zeros([(NX-bound)* NZ])
#             alpha = np.zeros(iter_i)
#             beta = np.zeros(iter_i)

#             ro=np.zeros(iter_i)
#             for iter_ii in range(iter_i):
#                 print(iter_ii,sm[iter_ii],ym[iter_ii],flush=True)
#                 ro[iter_ii]=1 / (np.dot(ym[iter_ii] , sm[iter_ii])+0.0000001)
#                 # ro[iter_ii]=1 / (np.dot(ym[iter_ii]* sm[iter_ii])+0.0000001)

#             print('ro',ro,flush=True)

#             q[iter_i]=delta_grad[PML + bound:-PML, PML:-PML].copy().reshape([(NX-bound)*NZ])

#             for iter_ii in range(iter_i-1,-1,-1):
#                 alpha[iter_ii]=ro[iter_ii]*np.dot(sm[iter_ii],q[iter_ii+1])
#                 # alpha[iter_ii]=ro[iter_ii]*np.sum(sm[iter_ii]*q[iter_ii+1])

#                 q[iter_ii]=q[iter_ii+1] - (alpha[iter_ii]*ym[iter_ii])

#             r=hdiag*q[0]

#             for iter_ii in range(iter_i):
#                 beta[iter_ii]=ro[iter_ii]*np.dot(ym[iter_ii],r)
#                 # beta[iter_ii]=ro[iter_ii]*np.sum(ym[iter_ii]*r)

#                 r=r+ sm[iter_ii]*(alpha[iter_ii]-beta[iter_ii])
#             print('beta',beta,flush=True)
#             print('alpha', alpha, flush=True)

#         else:
#             temp_sm=sm.copy()
#             temp_ym=ym.copy()
#             sm[:lt-1]=temp_sm[1:]
#             ym[:lt - 1] = temp_ym[1:]

#             sm[-1]=s0
#             ym[-1] = y0


#             q = np.zeros([lt + 1, (NX-bound)* NZ])
#             r = np.zeros([(NX-bound)*NZ])
#             alpha = np.zeros(lt)
#             beta = np.zeros(lt)

#             ro = np.zeros(lt)
#             for iter_ii in range(lt):
#                 ro[iter_ii] =1 / (np.dot(ym[iter_ii] , sm[iter_ii])+0.0000001)
#                 # ro[iter_ii] =1 / (np.sum(ym[iter_ii]* sm[iter_ii])+0.0000001)

#             q[lt] = delta_grad[PML + bound:-PML, PML:-PML].copy().reshape([(NX-bound)*NZ])

#             for iter_ii in range(lt - 1, -1, -1):
#                 alpha[iter_ii] = ro[iter_ii] * np.dot(sm[iter_ii],q[iter_ii+1])
#                 # alpha[iter_ii] = ro[iter_ii] * np.sum(sm[iter_ii]*q[iter_ii+1])

#                 q[iter_ii] = q[iter_ii + 1] - (alpha[iter_ii] * ym[iter_ii])

#             r = hdiag * q[0]

#             for iter_ii in range(lt):
#                 beta[iter_ii] = ro[iter_ii]*np.dot(ym[iter_ii],r)
#                 # beta[iter_ii] = ro[iter_ii]*np.sum(ym[iter_ii]*r)

#                 r = r + sm[iter_ii] * (alpha[iter_ii] - beta[iter_ii])


#         grad_pre = delta_grad.copy()
#         # if iter_i!=0:
#         delta_grad[PML + bound:-PML, PML:-PML] = r.reshape([(NX-bound),NZ])
#         grad_min, max_grad = np.percentile(abs(delta_grad[PML + bound:-PML, PML:-PML]), [2, 99])
#         delta_grad[PML + bound:-PML, PML:-PML] = delta_grad[PML + bound:-PML, PML:-PML] / max_grad

        if   iter_i<250:
            delta_grad=scipy.ndimage.filters.gaussian_filter(delta_grad, sigma=1)
#         else:
#             delta_grad=scipy.ndimage.filters.gaussian_filter(delta_grad, sigma=0.5)
        
        vp1=vp.copy()
        vp2=vp.copy()

        vp1=vp1+delta_1 * delta_grad
        vp2=vp2+delta_2 * delta_grad
        vp1 = vp1.astype(np.float32)
        vp2 = vp2.astype(np.float32)
    comm.Barrier()
    comm.Bcast(delta_grad, root=0)
    # print(delta_grad.shape, rank, flush=True)
    comm.Barrier()
    comm.Bcast(vp1, root=0)
    comm.Barrier()
    comm.Bcast(vp2, root=0)
    comm.Barrier()

    # print(vp2.shape,rank, flush=True)
        # vp += 50 * delta_grad
        # # plt.imshow(vp)
        # # plt.show()
        # grad_T[PML:-PML, PML:-PML].T.astype(np.float32).tofile(resultfile + str(iter_i) + 'grad.dat')
        # vp[PML:-PML, PML:-PML].T.astype(np.float32).tofile(resultfile + str(iter_i) + 'final_V.dat')
        # # print(time.time() - start)
        # np.savetxt(resultfile + "d_loss.txt", np.array(obj_f_list))
    comm.Barrier()
    comm.Bcast(vp1, root=0)
    comm.Bcast(vp2, root=0)
    # print(88888888888888888, flush=True)
    comm.Barrier()
    obj_f_1=0
    ddx, ddz = pml_ceof(vp1, NX_PML, NZ_PML, R, PML, DX, DZ, dt)
    # print(99999999999999999, flush=True)


    for si in range(int(np.ceil((len(sx_list)+1)/totalrank))):

        s_sum = rank+si*totalrank
        # print('rank: ',rank,'and s_sum: ', s_sum,flush = True)
        if s_sum<len(sx_list):
            # while (s_sum <3):
            # for s_num in range(3):
            #     if rank==( s_sum % 3 +1):
            obj_f_1 = 0
            # ddx, ddz = pml_ceof(vp1, NX_PML, NZ_PML, R, PML, DX, DZ, dt)
            import time
            # print(s_sum, rank,333333333333)
            record = np.zeros([NR, Tn])  ##地震记录
            """初始化波场参数"""
            P_now = np.zeros([NX_PML, NZ_PML])
            P_past = np.zeros([NX_PML, NZ_PML])
            P_next = np.zeros([NX_PML, NZ_PML])
            pt1 = np.zeros([NX_PML, NZ_PML])
            pt2 = np.zeros([NX_PML, NZ_PML])
            pv1 = np.zeros([NX_PML, NZ_PML])
            pv2 = np.zeros([NX_PML, NZ_PML])
            P_x = np.zeros([NX_PML, NZ_PML])  # vx
            P_z = np.zeros([NX_PML, NZ_PML])  # vz
            PP_tt = np.zeros([NX_PML, NZ_PML])
            # bochang = np.zeros([Tn, NX_PML, NZ_PML])
            for k in range(Tn):
                # print(k,flush = True)
                PP_tt, P_now, P_past, P_next, P_x, P_z, pv1, pv2, pt1, pt2 = cal(PP_tt, P_now, P_past, P_next, P_x, P_z,
                                                                                 pv1, pv2, pt1, pt2, ddx,
                                                                                 ddz, vp1, a, wave, NX_PML, NZ_PML,
                                                                                 sx_list[s_sum], sz_list[s_sum], k,DX,DZ,dt)

                for i in range(NR):
                    record[i, k] = P_now[rx_list[i], rz_list[i]]

                # PP_tt.tofile(bochangfile + str(k) + '.dat')
                # bochang[k] = PP_tt

            try:
                record_raw = np.fromfile(recordfile + str(s_sum) + '.dat', dtype=np.float32).reshape([NR, Tn])
            except:
                record_raw = np.fromfile(recordfile + str(s_sum) + '.bin', dtype=np.float32).reshape([NR, Tn])


            if fenpin == 1:
                for i_for_f in range(1, len(iter_f_list)):
                    if iter_i - iter_f_list[i_for_f - 1] >= 0 and iter_i - iter_f_list[i_for_f] < 0:
                        aa, bb = scipy.signal.butter(N_filter, 2 * f_list[i_for_f - 1] / (1 / dt), btype='lowpass',
                                                     analog=False,
                                                     output='ba',
                                                     fs=None)  # low_freq/ Nyquist freq == low_freq/（dt/2）
                        record = scipy.signal.filtfilt(aa, bb, record)
                        record_raw = scipy.signal.filtfilt(aa, bb, record_raw)
                
            # f_dt = 1e-3
            # f_nt = 4096
#             high_freq = int(3 / (1 / f_dt / f_nt))
#             fp = fft(record_raw)
#             fp[:, high_freq:-high_freq] = 0
#             record_raw = ifft(fp)
#             record_raw=record_raw.astype(np.float32)
            
#             fp = fft(record)
#             fp[:, high_freq:-high_freq] = 0
#             record = ifft(fp)
#             record=record.astype(np.float32)
            
            res = record - record_raw

            res = res.astype(np.float32)

            obj_f_1 += np.sum(res ** 2)

    comm.Barrier()
    obj_f_1 = comm.reduce(obj_f_1, root=0, op=MPI.SUM)
    # if rank==0:
    #     print(obj_f_1,'obf_1_j',rank,flush = True)

    # comm.Barrier()
    obj_f_2=0
    ddx, ddz = pml_ceof(vp2, NX_PML, NZ_PML, R, PML, DX, DZ, dt)

    for si in range(int(np.ceil((len(sx_list)+1)/totalrank))):
        # print(si+"333333333333333",flush = True)
        s_sum = rank+si*totalrank
        # print('rank: ',rank,'and s_sum: ', s_sum,flush = True)
        if s_sum<len(sx_list):
            # while (s_sum <3):
            # for s_num in range(3):
            #     if rank==( s_sum % 3 +1):
            obj_f_2 = 0
            # ddx, ddz = pml_ceof(vp1, NX_PML, NZ_PML, R, PML, DX, DZ, dt)
            import time
            # print(s_sum, rank,333333333333)
            record = np.zeros([NR, Tn])  ##地震记录
            """初始化波场参数"""
            P_now = np.zeros([NX_PML, NZ_PML])
            P_past = np.zeros([NX_PML, NZ_PML])
            P_next = np.zeros([NX_PML, NZ_PML])
            pt1 = np.zeros([NX_PML, NZ_PML])
            pt2 = np.zeros([NX_PML, NZ_PML])
            pv1 = np.zeros([NX_PML, NZ_PML])
            pv2 = np.zeros([NX_PML, NZ_PML])
            P_x = np.zeros([NX_PML, NZ_PML])  # vx
            P_z = np.zeros([NX_PML, NZ_PML])  # vz
            PP_tt = np.zeros([NX_PML, NZ_PML])
            # bochang = np.zeros([Tn, NX_PML, NZ_PML])
            for k in range(Tn):
                # print(k,flush = True)
                PP_tt, P_now, P_past, P_next, P_x, P_z, pv1, pv2, pt1, pt2 = cal(PP_tt, P_now, P_past, P_next, P_x, P_z,
                                                                                 pv1, pv2, pt1, pt2, ddx,
                                                                                 ddz, vp2, a, wave, NX_PML, NZ_PML,
                                                                                 sx_list[s_sum], sz_list[s_sum], k,DX,DZ,dt)

                for i in range(NR):
                    record[i, k] = P_now[rx_list[i], rz_list[i]]

                # PP_tt.tofile(bochangfile + str(k) + '.dat')
                # bochang[k] = PP_tt

            try:
                record_raw = np.fromfile(recordfile + str(s_sum) + '.dat', dtype=np.float32).reshape([NR, Tn])
            except:
                record_raw = np.fromfile(recordfile + str(s_sum) + '.bin', dtype=np.float32).reshape([NR, Tn])


            if fenpin == 1:
                for i_for_f in range(1, len(iter_f_list)):
                    if iter_i - iter_f_list[i_for_f - 1] >= 0 and iter_i - iter_f_list[i_for_f] < 0:
                        aa, bb = scipy.signal.butter(N_filter, 2 * f_list[i_for_f - 1] / (1 / dt), btype='lowpass',
                                                     analog=False,
                                                     output='ba',
                                                     fs=None)  # low_freq/ Nyquist freq == low_freq/（dt/2）
                        record = scipy.signal.filtfilt(aa, bb, record)
                        record_raw = scipy.signal.filtfilt(aa, bb, record_raw)
            # f_dt = 1e-3
            # f_nt = Tn
#             high_freq = int(3 / (1 / f_dt / f_nt))
#             fp = fft(record_raw)
#             fp[:, high_freq:-high_freq] = 0
#             record_raw = ifft(fp)
#             record_raw=record_raw.astype(np.float32)
            
#             fp = fft(record)
#             fp[:, high_freq:-high_freq] = 0
#             record = ifft(fp)
#             record=record.astype(np.float32)
            
            res = record - record_raw
            

            res = res.astype(np.float32)

            obj_f_2 += np.sum(res ** 2)
    comm.Barrier()

    obj_f_2 = comm.reduce(obj_f_2, root=0, op=MPI.SUM)

    if rank == 0:
        # print(44444444444444,flush=True)
        # grad_T = g / g_
        #
        #     print(obj_f,rank)
        # print(obj_f_2, 'obf_2_j', rank,flush = True)
        obj_f_list.append(obj_f)
        print(obj_f, 'obf', rank,flush = True)
        # print(4564564, flush=True)
        # delta_grad = np.zeros([NX_PML, NZ_PML])
        # max_grad = np.max(abs(grad_T[PML + 20:-PML, PML:-PML]))
        # delta_grad[PML + 20:-PML, PML:-PML] = grad_T[PML + 20:-PML, PML:-PML] / max_grad
        # comm.Bcast(delta_grad, root=0)
        b_=(delta_2*delta_2*obj_f_1-delta_1*delta_1*obj_f_2-obj_f*(delta_2*delta_2-delta_1*delta_1))/(delta_1*delta_2*(delta_2-delta_1))
        a_=(obj_f_1-obj_f-b_*delta_1)/(delta_1*delta_1)
        delta=-b_/(2*a_)
        if delta>100:
            delta=100
        elif delta < 0 and obj_f_2 >obj_f_1 and  obj_f_1 > obj_f:
            delta = 1
        elif delta > delta_2 and obj_f_2 >obj_f_1 and  obj_f_1 > obj_f:
            delta = 1
        elif delta < 0 and obj_f_2 < obj_f_1 and  obj_f_1 < obj_f:
            delta=delta_2
        elif delta < 0:
            delta=1

        # delta=10*(2-iter_i/50)
        # delta=10
        vp_pre=vp.copy()
        
        prior_m_regular=vp_prior[:, :]-vp[PML:-PML, PML:-PML]
        weight_prior = np.loadtxt("afwi_weight_over.txt").reshape([-1,NZ])
        prior_m_regular=prior_m_regular*weight_prior
        
        prior_m_regular=prior_m_regular/prior_m_regular.max()
        
        vp[PML:-PML, PML:-PML] += (delta * delta_grad[PML:-PML, PML:-PML]+(delta/10)*prior_m_regular)
        
#         vp += delta * delta_grad

        for i in range(vp.shape[0]):
            for j in range(vp.shape[1]):
                if vp[i,j]>5000.0:
                    vp[i, j] = 5000.0

        print(delta,'buchang',flush=True)

        if rank == 0:
            import numpy as np


#             all_record_0 = np.zeros([NS, NR, Tn])
#             for i in range(NS):
#                 # data_s = np.fromfile(record_dir + str(i) + '.bin', dtype=np.float32).reshape([450, 4096])
#                 try:
#                     data_s = np.fromfile(recordfile + str(i) + '.bin', dtype=np.float32).reshape([NR, Tn])
#                 except:
#                     data_s = np.fromfile(recordfile + str(i) + '.dat', dtype=np.float32).reshape([NR, Tn])
                    


#                 all_record_0[i, :, :] = data_s[:, :]


#             n_cmp = n_cmp
#             cmp_8_data = np.zeros([NR, n_cmp, Tn + NS + NR])  # 中心点覆盖次数为8的cmp道集

#             import os

#             # cmp_dir = "cmp_number_mar160/"
#             cmp_dir = "cmp_number/"

#             if not os.path.exists(cmp_dir):
#                 os.mkdir(cmp_dir)



#             n_450 = []
#             for i in range(0, NR):

#                 ar = np.loadtxt(cmp_dir + str(i) + "_cmp.txt")
#                 ar = ar.reshape([-1, 3])
#                 # print(ar)
#                 if ar.shape[0] != 0:
#                     # print(i)

#                     cmp_n = np.zeros([ar.shape[0], Tn + NS + NR])  # 求单个中心点的道集
#                     # print(ar.shape[0])
#                     for n in range(ar.shape[0]):

#                         # print(df_0)
#                         n_recv = ar[n, 0]
#                         n_shot = ar[n, 1] 
#                         n_recv = ar[n, 0]
#                         n_shot = ar[n, 1]
#                         # print(n_shot,n_recv)
#                         weight_d = np.arange(0, NS)
#                         weight_d = weight_d.astype(np.float32)
#                         train_indecies = [int(n_shot / delta_s)]
#                         train_indecies = np.array(train_indecies)
#                         source_coding = np.zeros(NS)

#                         for ii in range(weight_d.shape[0]):

#                             wd = train_indecies - weight_d[ii]
#                             wd = np.abs(wd)
#                             wd = wd
#                             # print(ww)
#                             for mm in range(wd.shape[0]):
#                                 if wd[mm] != 0:
#                                     wd[mm] = 1 / wd[mm]
#     #                                 wd[mm] = 1 

#                                 else:
#                                     wd[mm] = 1
#                             source_coding[ii] = np.sum(wd)

#                         source_coding = source_coding / source_coding.max()

#                         weight_d = np.arange(0, NR)
#                         weight_d = weight_d.astype(np.float32)
#                         train_indecies = [n_recv]
#                         train_indecies = np.array(train_indecies)
#                         receiv_coding = np.zeros(NR)

#                         for ii in range(weight_d.shape[0]):
#                             wd = train_indecies - weight_d[ii]
#                             wd = np.abs(wd)
#                             wd = wd
#                             # print(ww)
#                             for mm in range(wd.shape[0]):
#                                 if wd[mm] != 0:
#                                     wd[mm] = 1 / wd[mm]
#     #                                 wd[mm] = 1 

#                                 else:
#                                     wd[mm] = 1
#                             receiv_coding[ii] = np.sum(wd)

#                         receiv_coding = receiv_coding / receiv_coding.max()
                        
                        
#                         # print(n_shot,n_recv)

#                         if n < int(ar.shape[0] - 1) / 2:

#                             cmp_n[2 * n + 1, :Tn] = all_record_0[int(n_shot / delta_s), int(n_recv), :]
#                             cmp_n[2 * n + 1, Tn:Tn + NS] = source_coding[:]
#                             cmp_n[2 * n + 1, Tn + NS:Tn + NS + NR] = receiv_coding[:]


#                         else:

                            
#                             cmp_n[((ar.shape[0] - n - 1) * 2), :Tn] = all_record_0[int(n_shot / delta_s), int(n_recv), :]
#                             cmp_n[((ar.shape[0] - n - 1) * 2), Tn:Tn + NS] = source_coding[:]
#                             cmp_n[((ar.shape[0] - n - 1) * 2), Tn + NS:Tn + NS + NR] = receiv_coding[:]
                            

#                         # cmp_n[n]=all_record[int(n_shot/10),int(n_recv)] #求单个中心点的道集

#                         # cmp_stack[i]+=all_record[int(n_shot/10),int(n_recv)]  #求叠加道集
#                         # cmp_stack[i]=cmp_stack[i]/ar.shape[0]     #求叠加道集

#                     # cmp_n.astype(np.float32).tofile('cmp_'+str(i)+'_stack_'+str(n)+'_4096.bin') #求单个中心点的道集
#                     if ar.shape[0] > n_cmp:
#                         cmp_8_data[i, :, :] = cmp_n[-n_cmp:, :]
#                     else:
#                         cmp_8_data[i, -ar.shape[0]:, :] = cmp_n[:, :]
#                         for k in range(n_cmp - ar.shape[0]):
#                             cmp_8_data[i, k, :] = cmp_n[0, :]

#                     # cmp_stack.astype(np.float32).tofile('cmp_stack_450_4096.bin')

# #             print(cmp_8_data.shape)
# #             t_start=1
# #             for n_trace in range(NR):
# #                 for tk in range(1,Tn):
# #                     if cmp_8_data[n_trace,-1,tk]-cmp_8_data[n_trace,-1,tk-1]>=0.01:
# #                         t_start=tk
# #                         break
# #                     break
# #             #     print(t_start)
# #                 for i_cmp in range(n_cmp):
# #                     cmp_8_data[n_trace,n_cmp-i_cmp-1,0:t_start+455+85*i_cmp]=0


# #             """test: to emplify the amplitude of the weak signal"""
# #             for i in range(seismic_data_raw.shape[0]):
# #                 for j in range(seismic_data_raw.shape[1]):
# #                     for k in range(seismic_data_raw.shape[2]):
# #                         if seismic_data_raw[i,j,k] >0:
# #                             seismic_data_raw[i, j, k]=1
# #                         else:
# #                             seismic_data_raw[i, j, k] = -1

#             seismic_data_raw = cmp_8_data.copy()

#             # print(seismic_data_raw.shape)

#             seismic_data = np.zeros([NZ, n_channel, n_cmp, NS + NR + Tn])

#             for i in range(n_channel):
#                 seismic_data[:, i, :, :] = seismic_data_raw[:, :, :]

#             for x_i in range(NZ):
#                 if x_i ==0:
#                     for jj in range(n_channel-2):
#                         seismic_data[x_i, jj, :, :] = seismic_data_raw[x_i, :, :]
#                     seismic_data[x_i, n_channel-2, :, :] = seismic_data_raw[x_i+1, :, :]
#                     seismic_data[x_i, n_channel-1, :, :] = seismic_data_raw[x_i+2, :, :]
#                 if x_i==1:
#                     seismic_data[x_i, 0, :, :] = seismic_data_raw[x_i-1, :, :]
#                     seismic_data[x_i, 1, :, :] = seismic_data_raw[x_i-1, :, :]
#                     for jj in range(2,n_channel-2):
#                         seismic_data[x_i, jj, :, :] = seismic_data_raw[x_i, :, :]
                    
#                     seismic_data[x_i, n_channel-2, :, :] = seismic_data_raw[x_i+1, :, :]
#                     seismic_data[x_i, n_channel-1, :, :] = seismic_data_raw[x_i+2, :, :]
#                 if x_i >=2 and x_i <=(NZ-3):
#                     seismic_data[x_i, 0, :, :] = seismic_data_raw[x_i-2, :, :]
#                     seismic_data[x_i, 1, :, :] = seismic_data_raw[x_i-1, :, :]
                    
#                     for jj in range(2,n_channel-2):
#                         seismic_data[x_i, jj, :, :] = seismic_data_raw[x_i, :, :]
                    
#                     seismic_data[x_i, n_channel-2, :, :] = seismic_data_raw[x_i+1, :, :]
#                     seismic_data[x_i, n_channel-1, :, :] = seismic_data_raw[x_i+2, :, :]
#                 if x_i==(NZ-2):
#                     seismic_data[x_i, 0, :, :] = seismic_data_raw[x_i-2, :, :]
#                     seismic_data[x_i, 1, :, :] = seismic_data_raw[x_i-1, :, :]
#                     for jj in range(2,n_channel-2):
#                         seismic_data[x_i, jj, :, :] = seismic_data_raw[x_i, :, :]
#                     seismic_data[x_i, n_channel-2, :, :] = seismic_data_raw[x_i+1, :, :]
#                     seismic_data[x_i, n_channel-1, :, :] = seismic_data_raw[x_i+1, :, :]
#                 if x_i == (NZ-1):
#                     seismic_data[x_i, 0, :, :] = seismic_data_raw[x_i-2, :, :]
#                     seismic_data[x_i, 1, :, :] = seismic_data_raw[x_i-1, :, :]
#                     for jj in range(2,n_channel):
#                         seismic_data[x_i, jj, :, :] = seismic_data_raw[x_i, :, :]


            unlabeled_m_raw=vp[PML:-PML, PML:-PML].T





            # well_loc = [60,190,350]

            unlabeled_model=unlabeled_m_raw.copy()
#             unlabeled_model = np.delete(unlabeled_m_raw, well_loc, axis=0)

            unlabel_initial_model=model_initial.copy()
#             unlabel_initial_model = np.delete(model_initial, well_loc, axis=0)

            unlabeled_delta_model=unlabeled_model-unlabel_initial_model


            labeled_model_raw = vp_real
            labeled_model = labeled_model_raw[well_loc, :]

            from numpy.fft import fft, ifft

#             f_dt = 1e-3
#             f_nt = 175
#             fp = fft(labeled_model)
#             high_freq = int(30 / (1 / f_dt / f_nt))
#             fp[:, high_freq:-high_freq] = 0
#             labeled_el_new = ifft(fp)
#             labeled_el_new = labeled_el_new.astype(np.float32)
#             labeled_model[:,30:170]=labeled_el_new[:,30:170]

            """测井滤波"""
            log_vp1=labeled_model

            x_log=[ DX for i in range(NX)]

            x_log=np.array(x_log)

            t_log=x_log/log_vp1
            # print(t_log)
            dt_filter=np.mean(t_log)
            print(np.mean(t_log))

            if fenpin==1:
                for i_for_f in range(1, len(iter_f_list)):
                    if iter_i - iter_f_list[i_for_f - 1] >= 0 and iter_i - iter_f_list[i_for_f] < 0:
                        aa, bb = scipy.signal.butter(N_filter, 2 *(f_list[i_for_f - 1]+8) / (1 / dt_filter), btype='lowpass', analog=False, output='ba',fs=None)        
                        log2=labeled_model_raw[well_loc]
                        log2_filter = np.zeros([int(len(well_loc)),NX])
                        for log_num in range(int(len(well_loc))):
                                log2_filter[log_num,bound:] = scipy.signal.filtfilt(aa, bb, log2[log_num,bound:])
                        log2_filter[:,:bound]=log2[:,:bound]
                        labeled_model=log2_filter
            else:
                aa, bb = scipy.signal.butter(N_filter, 2 *(favg+8) / (1 / dt_filter), btype='lowpass', analog=False, output='ba',fs=None)        
                log2=labeled_model_raw[well_loc]
                log2_filter = np.zeros([int(len(well_loc)),NX])
                for log_num in range(int(len(well_loc))):
                        log2_filter[log_num,bound:] = scipy.signal.filtfilt(aa, bb, log2[log_num,bound:])
                log2_filter[:,:bound]=log2[:,:bound]
                labeled_model=log2_filter



            labeled_initial_model=model_initial[well_loc, :]

            labeled_model = labeled_model 
            labeled_initial_model = labeled_initial_model

            labeled_detla_model=labeled_model-labeled_initial_model




            unlabel_initial_model = unlabel_initial_model.reshape([-1, 1,1, NX])
            unlabeled_delta_model = unlabeled_delta_model.reshape([-1, 1, 1,NX])

            labeled_initial_model = labeled_initial_model.reshape([-1, 1,1, NX])
            labeled_detla_model = labeled_detla_model.reshape([-1, 1,1, NX])

            model_initial_net = model_initial.reshape([-1, 1, 1,NX])


            labeled_detla_model_max = np.max(labeled_detla_model, axis=(0, -1), keepdims=True)
            labeled_detla_model_min = np.min(labeled_detla_model, axis=(0, -1), keepdims=True)
            labeled_detla_model_mean = np.mean(labeled_detla_model, axis=(0, -1), keepdims=True)
            labeled_detla_model_std = np.std(labeled_detla_model, axis=(0, -1), keepdims=True)

            # labeled_detla_model = (labeled_detla_model - labeled_detla_model_mean) / (labeled_detla_model_std)#均值方差归一化
            labeled_detla_model = (labeled_detla_model - labeled_detla_model_min) / (labeled_detla_model_max-labeled_detla_model_min) #最大最小归一化

#             seis_mean = np.max(seismic_data[:, :, :, :Tn], axis=(0, -1), keepdims=True)
#             seis_std = np.min(seismic_data[:, :, :, :Tn], axis=(0, -1), keepdims=True)
#             seismic_data[:, :, :, :Tn] = (seismic_data[:, :, :, :Tn] - seis_std) / (seis_mean - seis_std)
#             seismic_data = seismic_data.reshape(NR, n_channel * n_cmp, Tn + NR + NS)
#             # print(seismic_data.shape)

#             seismic_data_temp = np.zeros([seismic_data.shape[0], seismic_data.shape[1], NX * 10]) #  设置增强地震道的长度  nx*10
#             seismic_data_temp[:, :, -(NR + NS):] = seismic_data[:, :, -(NR + NS):]
#             seismic_data_temp[:, :, :NX * 10 - (NR + NS)] = seismic_data[:, :, int((pi * 5 * 1.1) ** 2):int((pi * 5 * 1.1) ** 2)+NX * 10 - (NR + NS)] 
#             seismic_data = seismic_data_temp.copy()

                        
#             # print(seismic_data.shape)

#             unlabeled_seismic_data=seismic_data.copy()
# #             unlabeled_seismic_data = np.delete(seismic_data, well_loc, axis=0)

#             labeled_seismic_data = seismic_data[well_loc]

#             unlabeled_seismic_data = torch.tensor(unlabeled_seismic_data).float()
#             labeled_seismic_data = torch.tensor(labeled_seismic_data).float()
#             seismic_data = torch.tensor(seismic_data).float()
#             if torch.cuda.is_available():
#                 seismic_data=seismic_data.cuda()
#                 unlabeled_seismic_data=unlabeled_seismic_data.cuda()
#                 labeled_seismic_data=labeled_seismic_data.cuda()

            unlabeled_delta_model_mean = np.mean(unlabeled_delta_model, axis=(0, -1), keepdims=True)
            unlabeled_delta_model_std = np.std(unlabeled_delta_model, axis=(0, -1), keepdims=True)
            # unlabeled_delta_model = (unlabeled_delta_model - unlabeled_delta_model_mean) / (unlabeled_delta_model_std)#均值方差归一化
            unlabeled_delta_model=(unlabeled_delta_model-labeled_detla_model_min)/(labeled_detla_model_max-labeled_detla_model_min)#最大最小归一化


            unlabeled_delta_model_mean = torch.tensor(unlabeled_delta_model_mean).float()   #测试 对 unlabel 和label 分别做不同的归一化是否有不同效果
            unlabeled_delta_model_std = torch.tensor(unlabeled_delta_model_std).float()


            labeled_detla_model_max = torch.tensor(labeled_detla_model_max).float()
            labeled_detla_model_min = torch.tensor(labeled_detla_model_min).float()


            labeled_detla_model = torch.tensor(labeled_detla_model).float()
            unlabeled_delta_model = torch.tensor(unlabeled_delta_model).float()
            labeled_initial_model = torch.tensor(labeled_initial_model).float()
            unlabel_initial_model = torch.tensor(unlabel_initial_model).float()

            labeled_detla_model_mean_net = torch.tensor(labeled_detla_model_mean).float()
            labeled_detla_model_std_net = torch.tensor(labeled_detla_model_std).float()

            model_initial_net=torch.tensor(model_initial_net).float()






            labeled_model = torch.tensor(labeled_model).float()


            weight_net = np.loadtxt("afwi_weight_over.txt")
            weight_net=weight_net.reshape([NZ])
            weight_net = torch.tensor(weight_net).float()
            
            
            
            snr_torch=torch.tensor(SNR / 10)
            if torch.cuda.is_available():
                weight_net=weight_net.cuda()
                labeled_detla_model_max=labeled_detla_model_max.cuda()
                labeled_detla_model_min=labeled_detla_model_min.cuda()

                model_initial_net=model_initial_net.cuda()
                unlabeled_delta_model=unlabeled_delta_model.cuda()
                labeled_detla_model=labeled_detla_model.cuda()
                labeled_initial_model=labeled_initial_model.cuda()
                labeled_initial_model=labeled_initial_model.cuda()
                labeled_detla_model_mean_net=labeled_detla_model_mean_net.cuda()
                labeled_detla_model_std_net=labeled_detla_model_std_net.cuda()
                labeled_model=labeled_model.cuda()
                unlabeled_delta_model_mean =unlabeled_delta_model_mean.cuda()
                unlabeled_delta_model_std=unlabeled_delta_model_std.cuda()
                wavelet_noise=wavelet_noise.float().cuda()
                snr_torch=snr_torch.cuda()


            #
            # print("unlabeled_seismic_data:",unlabeled_seismic_data,flush=True)
            # print("unlabeled_delta_model:",unlabeled_delta_model,flush=True)

            # unlabeled_loader = data.DataLoader(
            #     data.TensorDataset(unlabeled_seismic_data, unlabeled_delta_model,weight),
            #     batch_size=batch_size,
            #     shuffle=True)
            unlabeled_loader = data.DataLoader(
                data.TensorDataset(unlabeled_seismic_data, unlabeled_delta_model,weight_net),
                batch_size=batch_size,
                shuffle=True)
            inverse_net = Seis_UnetModelDropout(n_classes=1, in_channels=n_cmp * n_channel, NX=NX, is_deconv=True,
                                        is_batchnorm=True)
            gen_net = Generator()
            if torch.cuda.is_available():
                inverse_net.cuda()
                gen_net.cuda()
            gen_net.train()
            inverse_net.train()
            criterion = nn.MSELoss()

            # criterion = nn.SmoothL1Loss()
            #
#             optimizer = inverse_net.optimizer

            optimizer = torch.optim.Adam([
                {'params': inverse_net.parameters(), 'lr': 0.003,'weight_decay ':1e-4}
            ])
            optimizer_gen = torch.optim.Adam([
                {'params': gen_net.parameters(), 'lr': 0.003,'weight_decay ':1e-4}
            ])
            print("Training the model")
            best_loss = np.inf
            train_loss = []
            train_property_corr = []
            train_property_r2 = []
            if iter_i%1==0:
                for epoch in tqdm(range(300)):
                    for x, y,weight in unlabeled_loader:
                        optimizer.zero_grad()

                        label_x_2D_temp = labeled_seismic_data
                        print(label_x_2D_temp.size(), 'label_x_2D_temp')
                        label_x_2D_temp_std = torch.std(label_x_2D_temp * noise_para)

                        label_x_2D_temp += (label_x_2D_temp_std ** 0.5) * torch.randn_like(label_x_2D_temp)
                        if SNR!=0:
                            label_x_2D_temp=add_noise_torch(labeled_seismic_data,wavelet_noise,snr_torch=snr_torch,Tn=NX * 10)
                        y_pred_delta_label = inverse_net(label_x_2D_temp, label_dsp_dim)
                        # print(y_pred_delta_label.size(), 'y_pred_delta_label')

                        property_loss = criterion(y_pred_delta_label, labeled_detla_model)


                        x_2D_temp = x

                        x_2D_temp_std = torch.std(x * noise_para)

                        x_2D_temp += (x_2D_temp_std ** 0.5) * torch.randn_like(x_2D_temp)
                        if SNR!=0:
                            x_2D_temp=add_noise_torch(x,wavelet_noise,snr_torch=snr_torch,Tn=NX * 10)

                        y_pred = inverse_net(x_2D_temp, label_dsp_dim)
                        weight_in_net = (1 - weight) / weight
                        weight_in_net = weight_in_net.reshape([-1, 1, 1, 1])
                        #                     print(y_pred.size())
                        #                     print(weight_in_net.size())
                        property_loss_2 = criterion(weight_in_net * y_pred, weight_in_net * y)
                        # property_loss_2 = criterion(y_pred, y)

                        # property_loss_2 = criterion(y_pred, y)
                        # w_m = torch.mean(w_1)
                        # loss = (w_m/(p-w_m))* property_loss / property_loss.detach().clone() + 15*property_loss_2 / property_loss_2.detach().clone()
    #                     loss = 0.01*property_loss + 1000*property_loss_2
                        if iter_i>60:
                            loss = (1.0)*property_loss / property_loss.detach().clone() + (10)*property_loss_2 / property_loss_2.detach().clone()
                        else:
                            loss = (1)*property_loss / property_loss.detach().clone() + (100)*property_loss_2 / property_loss_2.detach().clone()
    #                     loss = (1.0)*property_loss / property_loss.detach().clone() + (100.0-iter_i)*property_loss_2 / property_loss_2.detach().clone()
    #                     loss = 1.0*property_loss + 100*property_loss_2
#                         if iter_i>60:
#                             loss = (1.0)*property_loss / property_loss.detach().clone() + (100)*property_loss_2 / property_loss_2.detach().clone()
#                         else:
#                             loss = (1)*property_loss / property_loss.detach().clone() + (10)*property_loss_2 / property_loss_2.detach().clone()
    
                        loss.backward()
                        optimizer.step()

                        train_loss.append(loss.detach().clone())
                    print(" property_loss = {} ".format(property_loss.detach().clone()),flush=True)
                    print(" property_loss2 = {} ".format(property_loss_2.detach().clone()),flush=True)
                    print(" loss = {}".format(loss.detach().clone()))

                save_dir = "model_dir/"
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                session_name = 'rnn'
                torch.save(inverse_net, save_dir + '{}'.format(session_name))
                # sio.savemat(args.save_dir+'/'+"save_name"+'_train_loss.mat', {'train_loss': train_loss})
                # loss_arr=np.array(train_loss)
                # np.savetxt(args.save_dir+'/'+'train_loss.txt',loss_arr)

                test_loader = data.DataLoader(
                    data.TensorDataset(seismic_data,model_initial_net),
                    batch_size=batch_size,
                    shuffle=False)
                result_list = []
                for _  in range(10):
                    predicted_impedance = []
                    for x,initial_m_net in test_loader:
                        optimizer.zero_grad()

                        x_2D_temp = x
                        if SNR!=0:
                            x_2D_temp=add_noise_torch(x,wavelet_noise,snr_torch=snr_torch,Tn=NX * 10)

                        x_2D_temp_std = torch.std(x * noise_para)

                        x_2D_temp += (x_2D_temp_std ** 0.5) * torch.randn_like(x_2D_temp)
                        y_pred = inverse_net( x_2D_temp,label_dsp_dim)
                        # print(y_pred.size())
                        # y_pred=y_pred*2500-1200
                        # y_pred = y_pred * labeled_detla_model_std_net + labeled_detla_model_mean_net
                        y_pred = y_pred * (labeled_detla_model_max-labeled_detla_model_min)+labeled_detla_model_min
    #                     y_pred=y_pred*unlabeled_delta_model_std+unlabeled_delta_model_mean
                        y_pred=y_pred+initial_m_net

                        predicted_impedance.append(y_pred)
                    predicted_impedance = torch.cat(predicted_impedance, dim=0)

                    predicted_impedance = predicted_impedance.cpu().detach().numpy()
                    result_list.append(predicted_impedance)
                result_list = np.array(result_list)
                result_mean = np.mean(result_list, axis=(0))
                result_std = np.std(result_list, axis=(0))

                # print(predicted_impedance.shape)
                # predicted_impedance = predicted_impedance * labled_v_std + labeled_v_mean
                result_mean=result_mean.reshape([NZ, NX])

                result_mean.astype(np.float32).tofile(resultfile + str(iter_i) + 'middle_result_v.bin')
            else:
                result_mean=vp[PML :-PML,PML :-PML].T

            result_delta = (result_mean - model_initial) 
            delta_min=np.min(result_delta)
            delta_max=np.max(result_delta)
            result_delta=(result_delta-delta_min)/(delta_max-delta_min)
            
            grad_label = torch.Tensor(result_delta[:, :].T)
            grad_label = grad_label.to(device)

            # criterion = torch.nn.SmoothL1Loss()
            # net_model = CNN().to(device)
            for epoch in range(1000):
                # input = np.random.randn(1, 100, 1, 5)
                # print(epoch, flush=True)
                # input = np.ones([1, 100, 1, 5]) * 0.05
                input_data = np.random.random([1, 100, 1, 5])
                input_data = torch.Tensor(input_data)
                input_data = input_data.to(device)
                # print(epoch)
                optimizer_gen.zero_grad()
                grad_pred = gen_net(input_data, NX=NX, NZ=NZ, bound=0)
                # print(grad_pred.size())
                # print(model_true.size())

                # loss_model = criterion(model_pred, model_true-model_init)
                # loss_model = criterion(model_pred, model_init)
                loss_g = criterion(grad_pred, grad_label)

                loss_g.backward(retain_graph=True)
                optimizer_gen.step()

            result_list_2 = []
            for _ in range(10):
                optimizer_gen.zero_grad()

                # input = np.ones([1, 100, 1, 5]) * 0.05
                input_data = np.random.random([1, 100, 1, 5])

                input_data = torch.Tensor(input_data)
                input_data = input_data.to(device)
                # print(epoch)
                grad_pred = gen_net(input_data, NX=NX, NZ=NZ, bound=0)

                grad_pred = grad_pred.cpu().detach().numpy()
                result_list_2.append(grad_pred*(delta_max-delta_min)+delta_min+model_initial.T)
            result_list_2 = np.array(result_list_2)
            result_mean_from_gen = np.mean(result_list_2, axis=(0))
            result_std_from_gen = np.std(result_list_2, axis=(0))


#             result_mean_from_gen = result_mean_from_gen.T   #可以试试不对速度模型做平滑 但是对梯度做平滑
            
#             if   iter_i<80:
#                 result_mean_from_gen=scipy.ndimage.filters.gaussian_filter(result_mean_from_gen, sigma=1)
#             else:
#                 result_mean_from_gen=scipy.ndimage.filters.gaussian_filter(result_mean_from_gen, sigma=0.5)
#             if iter_i < 100:
#                 result_mean_from_gen=scipy.ndimage.filters.gaussian_filter(result_mean_from_gen, sigma=1)

#             if len(obj_f_list)>3:
#                 if obj_f_list[-1]/obj_f_list[0] - obj_f_list[-2]/obj_f_list[0] < 0.01:
#                     result_mean_from_gen=scipy.ndimage.filters.gaussian_filter(result_mean_from_gen, sigma=1)
#                 else:
#                     result_mean_from_gen=scipy.ndimage.filters.gaussian_filter(result_mean_from_gen, sigma=1)


            result_std_from_gen=result_std_from_gen.reshape([NZ, NX])
            result_std_from_gen = result_std_from_gen.T

            # vp[PML:-PML, PML+40:-PML-40]=predicted_impedance[:,40:-40]
            vp[PML+ bound:-PML, PML :-PML ] = result_mean_from_gen[bound:,:]
            test_back=0
            if len(obj_f_list)>2:
                test_back=(obj_f_list[-1]-obj_f_list[-2])/obj_f_list[0]
            print(test_back,"test_back",flush=True)
            if np.isnan(vp[int(NX/2),int(NZ/2)]) or test_back >0.05:
                vp=np.fromfile(resultfile+'{}final_V.bin'.format(str(iter_i-2)),np.float32).reshape([NZ,NX]) 
                vp[:,bound:]=scipy.ndimage.filters.gaussian_filter(vp[:,bound:], sigma=1)
                vp=vp.T
                vp = ext_model(vp, NX, NZ, PML)
            
            for i in range(vp.shape[0]):
                for j in range(vp.shape[1]):
                    if vp[i, j] > 5000.0:
                        vp[i, j] = 5000.0
                    if vp[i, j] < 1500.0:
                        vp[i, j] = 1500.0
        # comm.Barrier()
        # comm.Bcast(vp, root=0)
        # print(delta_grad.shape, rank, flush=True)
        # comm.Barrier()
    #     # plt.imshow(vp)
    #     # plt.show()
    #     grad_T[PML:-PML, PML:-PML].T.astype(np.float32).tofile(resultfile + str(iter_i) + '_orginal_grad.dat')
        delta_grad[PML:-PML, PML:-PML].T.astype(np.float32).tofile(resultfile + str(iter_i) + 'grad.bin')
        vp[PML:-PML, PML:-PML].T.astype(np.float32).tofile(resultfile + str(iter_i) + 'final_V.bin')
        result_std_from_gen.T.astype(np.float32).tofile(resultfile + str(iter_i) + 'std_V.bin')
        np.savetxt(resultfile + "d_loss.txt", np.array(obj_f_list))


    comm.Bcast(vp, root=0)
    comm.Barrier()
