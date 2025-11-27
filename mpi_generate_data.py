import numpy as np
import math

#
import time
from numba import jit
import os
from mpi4py import MPI
# import sys
comm = MPI.COMM_WORLD   #Communicator对象包含所有进程
# client_script = 'mpi_fwi.py'
# comm = MPI.COMM_SELF.Spawn(sys.executable, args=[client_script], maxprocs=3)
size = comm.Get_size()
rank = comm.Get_rank()
print('rank',rank,'size ',size,flush=True)

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

import scipy.signal

# if rank == 0:

DX = 20
DZ = 20
dt = 0.002
NX = 175
NZ = 450


PML = 30

NR = 450                       
NS = 45

delta_s = 10
delta_r = 1

NX_PML = NX + 2 * PML
NZ_PML = NZ + 2 * PML

favg = 10
Tn = 3000  # 时间长度

"""加噪音"""
SNR=0
""""切低频"""
lowcut=1
lowcut_f=2
if rank ==0:
    bochangfile = 'bochang_new/'
    if not os.path.exists(bochangfile):
        os.mkdir(bochangfile)
    recordfile='over_{}_{}_f_{}_dx_{}_snr_{}_lowcut{}Hz/'.format(NS,NR,favg,DX,SNR,lowcut_f)
#     recordfile='mar2_{}_{}_f_{}_dx_{}_snr_{}_lowcut{}Hz/'.format(NS,NR,favg,DX,SNR,lowcut_f)
    if not os.path.exists(recordfile):
        os.mkdir(recordfile)
        
        


# recordfile='mar2_{}_{}_f_{}_dx_{}_snr_{}_lowcut{}Hz/'.format(NS,NR,favg,DX,SNR,lowcut_f)
recordfile='over_{}_{}_f_{}_dx_{}_snr_{}_lowcut{}Hz/'.format(NS,NR,favg,DX,SNR,lowcut_f)

w = ricker(dt, favg, Tn)
w=w.astype(np.float32)

if lowcut:
    aa, bb = scipy.signal.butter(6, 2 * lowcut_f/ (1 / dt), btype='highpass',
                             analog=False, output='ba',
                             fs=None)  # low_freq/ Nyquist freq == low_freq/（dt/2）
    w = scipy.signal.filtfilt(aa, bb, w)


#########加相干噪声########
from bruges.filters import wavelets
wavelet_noise = wavelets.ricker(0.8, dt, favg)
wavelet_noise=wavelet_noise/wavelet_noise.max()
def convlution(data,wavelet):
    data_new=np.zeros(data.shape[0]+wavelet.shape[0])
    data_cov=np.zeros(data.shape[0]+wavelet.shape[0])
    data_cov[:data.shape[0]]=data[:]
    data_cov[data.shape[0]:]=np.random.normal(0, 1, wavelet.shape)
    for i in range(0,data_cov.shape[0]-wavelet.shape[0]):
            if i <=data.shape[0]:
                a= data_cov[i:i+wavelet.shape[0]]*wavelet[:]
                data_new[i]=np.sum(a)
            elif i<data_cov.shape[0]:
                a= data_cov[i:]*wavelet[i-data.shape[0]:]
                data_new[i]=np.sum(a)
    return data_new[:data.shape[0]]


"""带通滤波"""
# low_freq=int(3/(1/dt/Tn))
# high_freq = int(50/(1/dt/Tn))

# b = split_f(w.reshape([1, -1]), low_f=low_freq, high_f=high_freq)
# w=b[0].astype(np.float32)

"""高通滤波"""
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



"""定义速度和密度模型"""
# vp = np.ones([NX, NZ]) * 3000
# vp[150:, :] = 5000
# den = np.ones([NX, NZ]) * 3000
# den[150:, :] = 4000
# model = np.fromfile("marmousi_450_3_175.bin", dtype=np.float32).reshape([NZ,3, NX])
# vp=model[:,0,:]
# vp = np.fromfile("marmousi_450_3_175.bin", dtype=np.float32).reshape([NZ, NX])

vp = np.fromfile("overthrust_450_175.bin", dtype=np.float32).reshape([NZ, NX])


# vp = np.fromfile("marmousi_vp_450_175_linear_1500_4500.bin", dtype=np.float32).reshape([NZ, NX])
# vp = np.fromfile("nncfwi_result_f_5_snr0.bin", dtype=np.float32).reshape([NZ, NX])
# vp = np.fromfile("cmp_result_f_3_no_lowf_450_175_snr0.bin", dtype=np.float32).reshape([NZ, NX])
bound=23
# vp[:,:bound]=model[:,0,:bound]
vp = vp.T


vp = ext_model(vp, NX, NZ, PML)
print(vp.shape)
# vp=vp.reshape([1,-1])
# vp=np.squeeze(vp)
# den = ext_model(den, NX, NZ, PML)
obj_f_list = []
# sx_list = [int(np.ceil(PML + 5)) for i in range(0, NS * delta_s, delta_s)]
# sz_list = [int(np.ceil(PML + i)) for i in range(0, NS * delta_s, delta_s)]  # 震源深度  每一炮

# rx_list = [int(np.ceil(PML + 8)) for i in range(0, NR * delta_r, delta_r)]
# rz_list = [int(np.ceil(PML + i)) for i in range(0, NR * delta_r, delta_r)]  # 接收点深度  每个点都接收
sx_list = [int(np.ceil(PML + 0)) for i in range(0, NS * delta_s, delta_s)]
sz_list = [int(np.ceil(PML + i)) for i in range(0, NS * delta_s, delta_s)]  # 震源深度  每一炮

rx_list = [int(np.ceil(PML + 0)) for i in range(0, NR * delta_r, delta_r)]
rz_list = [int(np.ceil(PML + i)) for i in range(0, NR * delta_r, delta_r)]  # 接收点深度  每个点都接收
a = diff_coef(N)

R = 1e-10


iter_i=0

if rank==0:
    obj_f = 0
    obj_f_list = []
obj_f = comm.bcast(obj_f if rank == 0   else None, root=0)

comm.Barrier()
# print(11111,rank)
vp1= np.zeros([NX_PML, NZ_PML],dtype=np.float32)
vp2= np.zeros([NX_PML, NZ_PML],dtype=np.float32)
delta_grad= np.zeros([NX_PML, NZ_PML],dtype=np.float32)
totalrank=size
for iter_i in range(1):
    comm.Barrier()
    print(iter_i,flush=True)
    ddx, ddz = pml_ceof(vp, NX_PML, NZ_PML, R, PML, DX, DZ, dt)
    g = np.zeros([NX_PML, NZ_PML])
    g_ = np.zeros([NX_PML, NZ_PML])
    for si in range(int(np.ceil((len(sx_list)+1)/totalrank))):
        # print(si,flush = True)
        s_sum = rank+si*totalrank
        if s_sum<len(sx_list):
            print('rank: ', rank, 'and s_sum: ', s_sum, sx_list[s_sum], sz_list[s_sum], flush=True)
            # while (s_sum <3):
            # for s_num in range(3):
            #     if rank==( s_sum % 3 +1):
            obj_f = 0
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
            # print(vp.shape,'vp\n',a.shape,'a\n',w.shape,'w\n',ddx.shape,'ddx\n',ddz.shape,'ddz\n',len(sx_list),'sx_list\n')
            for k in range(Tn):
                # print(k,flush = True)
                PP_tt, P_now, P_past, P_next, P_x, P_z, pv1, pv2, pt1, pt2 = cal(PP_tt, P_now, P_past, P_next, P_x, P_z,
                                                                                 pv1, pv2, pt1, pt2, ddx,
                                                                                 ddz, vp, a, w, NX_PML, NZ_PML,
                                                                                 sx_list[s_sum], sz_list[s_sum], k,DX,DZ,dt)

                for i in range(NR):
                    record[i, k] = P_now[rx_list[i], rz_list[i]]

            if SNR!=0:
#                 noise_temp = np.random.normal(0, 1, record.shape)
#                 ps = np.mean(np.power(record, 2))
#                 pn1 = np.mean(np.power(noise_temp, 2))
#                 k = np.sqrt(ps / (np.power(10, SNR / 10)))  #
#                 noise_data = noise_temp * k
#                 record = record + noise_data
                
                

                noise_temp = np.random.normal(0, 1, record.shape)
                noise_data=np.zeros(record.shape)
                for dao in range(NR):
                    noise_data[dao]=convlution(noise_temp[dao],wavelet_noise)
                ps = np.mean(np.power(record, 2))
                pn1 = np.mean(np.power(noise_data, 2))
                k = np.sqrt(ps / (np.power(10, SNR / 10)))  #
                noise_data_xianggan = noise_data * k
                record = record + noise_data_xianggan
#             if lowcut:
#                 aa, bb = scipy.signal.butter(6, 2 * lowcut_f/ (1 / dt), btype='highpass',
#                                          analog=False, output='ba',
#                                          fs=None)  # low_freq/ Nyquist freq == low_freq/（dt/2）
#                 record = scipy.signal.filtfilt(aa, bb, record)
            record.astype(np.float32).tofile(recordfile + str(s_sum) + '.bin')


    comm.Barrier()
comm.Barrier()



