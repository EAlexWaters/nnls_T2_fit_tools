# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 13:55:56 2021

@author: alex
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.util import montage
from scipy.optimize import nnls

def calc_t2_signal(TE,T2,S0):
    '''
    Parameters
    ----------
    TE : (float) Single echo time 
    T2 : (float) T2 value for which to compute signal 
    S0 : (float) Initial signal intensity 

    Returns
    -------
    (float) T2 weighted signal according to the Bloch equations 

    '''
    return S0 * np.exp(-TE/T2) 

def create_t2_matrix(TEs, T2s, S0s):
    '''
    Creates the Multi Echo T2 (spectrum) design matrix.
    Given a grid of echo times (numpy vector TEs) and a grid of T2 times
    (numpy vector T2s), it returns the deign matrix to perform the inversion.
    '''
    M = len(TEs)
    N = len(T2s)
    t2_matrix = np.zeros((M,N))
    for row in range(M):
        for col in range(N):
            t2_matrix[row,col] = calc_t2_signal(TEs[row],T2s[col],S0s[col])
        # end for col
    # end for row
    return t2_matrix

def save_scaled_montage(fn, img, max_val):
    this_montage = montage(img)
    this_montage = (this_montage/max_val)*255
    plt.imshow(this_montage.squeeze(), interpolation='nearest',vmax=max_val)
    plt.show()
    montage_name = fn+'_montage.png'
    imageio.imwrite(montage_name ,this_montage.astype('uint8'))
    

'''
Function: fit_t2_nnls
Input: a 4D image volume from an MRI T2 map acquisition with indices as: [slices, echoes, xdim, ydim]
Returns: a 4D array with indices as: [slices, xdim, ydim, t2vals]
'''
def fit_t2_nnls(vol, A_mat, num_t2s):
    
    nslices_in = vol.shape[0]
    nechoes_in = vol.shape[1]
    xdim = vol.shape[2]
    ydim = vol.shape[3]
    # Initialize output array
    t2_vol_arr_out = np.ndarray([nslices_in,xdim,ydim,num_t2s])
    
    # Iterate over slices
    for s in range(nslices_in):
        print("fitting slice " + str(s))
        
        # iterate over voxels in each slice
        for m in range(vol.shape[-1]):
            for n in range(vol.shape[-2]):
                
                # extract the array of measured T2 values for the current voxel
                t2data = np.squeeze(vol[s,:,m,n])
                
                # if the data is nonzero (i.e. not masked out)
                if t2data[0]:
                    
                    # perform the NNLS fit of measured data against the dictionary
                    try:
                        fit_result, rnorm = nnls(A_mat,t2data)
                        t2_vol_arr_out[s,m,n,:]=fit_result
                        
                    except RuntimeError:
                        t2data = -1
                else: t2data = 0
    return t2_vol_arr_out
    
    
def nnls_tik(A,b,L):
    C = np.concatenate((A,L),axis=0)
    D = np.concatenate((b,np.zeros(A.shape[1])))
    fit_result, rnorm = nnls(C,D)

    return fit_result    

'''
Function: fit_t2_nnls
Input: a 4D image volume from an MRI T2 map acquisition with indices as: [slices, echoes, xdim, ydim]
Returns: a 4D array with indices as: [slices, xdim, ydim, t2vals]
'''
def fit_t2_nnls_tik(vol, A_mat, num_t2s, reg_param, base_reg_mtx):
    
    num_param_vals = np.size(reg_param)
    # Initialize output array
    t2_vol_arr_out = np.array(num_t2s)
    
    
    summed_vol = np.sum(vol, axis = (0,2,3))
    avg_vol = summed_vol/(vol.shape[0]+vol.shape[2]+vol.shape[3] )
    
    resid_norm_arr = np.zeros(num_param_vals)
    reg_norm_arr = np.zeros(num_param_vals)
    for l in range(0,num_param_vals):              
        fit_result_tik = nnls_tik(A_mat,avg_vol,reg_param[l]*base_reg_mtx)
        
        reg_norm = np.dot(base_reg_mtx,fit_result_tik)
        reg_norm_arr[l] = np.dot(reg_norm.T,reg_norm)
        
        resid_norm = summed_vol - np.dot(A_mat,fit_result_tik)
        resid_norm_arr[l] = np.dot(resid_norm.T,resid_norm) 
        
    cte = np.cos(7*np.pi/8)
    cosmax = -2
    npt = num_param_vals
    corner = num_param_vals
    C = np.array([np.log(resid_norm_arr[-1]),np.log(reg_norm_arr[-1])])
    for k in range(npt-2):
        B = np.array([np.log(resid_norm_arr[k]),np.log(reg_norm_arr[k])])
        for j in range(k, npt-2):
            A = np.array([np.log(resid_norm_arr[k+1]),np.log(reg_norm_arr[k+1])])
            BA = B-A
            AC = A-C
            ATB = np.dot(BA,-1*AC)
            cosfi = ATB/(np.linalg.norm(BA)*np.linalg.norm(AC))
            #area = 0.5*np.linalg.det([BA,AC])
            area = 0.5*((B[0]-A[0])*(A[1]-C[1])-(A[0]-C[0])*(B[1]*B[0]))
            #print('cosfi and area: '+ str(cosfi) + ' ' + str(area))
            if ((cosfi > cte) and (cosfi > cosmax) and (area<0)):
                corner = j+1
                cosmax = cosfi
                print('updating corner to ' + str(corner))
                
    return nnls_tik(A_mat,summed_vol,corner*base_reg_mtx)
    