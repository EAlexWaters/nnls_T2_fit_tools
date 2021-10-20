# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:16:22 2020

@author: EAlex
"""
import imageio, os
import matplotlib.pyplot as plt
import numpy as np


import nnls_T2_functions as nnt2
import image_info_constants as ii


# Set working directory and path to DICOM folder

wd=r'C:\Users\alex\Box\DataCamp\NNLS_T2_fitting\TestData'
# bulk_storage_dir = r'C:\Users\alex\Box\DataCamp\NNLS_T2_fitting\TestData\NNLS_fits'
bulk_storage_dir = r"E:\Data_current\MRI\2021\McNally\McN202109_165A-170E_nnls_fits"


te_arr = np.linspace(ii.min_te,ii.max_te,ii.nechoes)

# set up array of T2 values to use in dictionary
lo_max = 40
mid_max = 200
t2_max = 1000
num_t2s = 120
t2_arr = np.logspace(np.log10(10),np.log10(t2_max),num_t2s)


lo_range = np.where(t2_arr < lo_max)
mid_range = np.where(np.logical_and(t2_arr > lo_max, t2_arr < mid_max))
hi_range = np.where(t2_arr > mid_max)

# build dictionary
A_mat = nnt2.create_t2_matrix(te_arr,t2_arr,np.ones(t2_arr.size))


# read masked T2 data

filename_list = (r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_165A_0927_LegT2\Analysis\nnls\165A_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_165B_0927_LegT2\Analysis\nnls\165B_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_165C_0927_LegT2\Analysis\nnls\165C_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_165D_0927_LegT2\Analysis\nnls\165D_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_165E_0927_LegT2\Analysis\nnls\165E_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_166A_0927_LegT2\Analysis\nnls\166A_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_166B_0927_LegT2\Analysis\nnls\166B_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_166C_0927_LegT2\Analysis\nnls\166C_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_166D_0928_LegT2\Analysis\nnls\166D_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_166E_0928_LegT2\Analysis\nnls\166E_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_167A_0928_LegT2\Analysis\nnls\167A_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_167B_0928_LegT2\Analysis\nnls\167B_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_167C_0928_LegT2\Analysis\nnls\167C_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_167D_0928_LegT2\Analysis\nnls\167D_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_167E_0928_LegT2\Analysis\nnls\167E_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_168A_0929_LegT2\Analysis\nnls\168A_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_168B_0929_LegT2\Analysis\nnls\168B_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_168C_0929_LegT2\Analysis\nnls\168C_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_168D_0929_LegT2\Analysis\nnls\168D_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_168E_0929_LegT2\Analysis\nnls\168E_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_169A_0929_LegT2\Analysis\nnls\169A_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_169B_0929_LegT2\Analysis\nnls\169B_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_169C_0929_LegT2\Analysis\nnls\169C_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_169D_0930_LegT2\Analysis\nnls\169D_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_MDX_169E_0930_LegT2\Analysis\nnls\169E_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_WT_170A_0927_LegT2\Analysis\nnls\170A_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_WT_170B_0928_LegT2\Analysis\nnls\170B_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_WT_170C_0929_LegT2\Analysis\nnls\170C_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_WT_170D_0930_LegT2\Analysis\nnls\170D_t2raw_masked.raw",
                 r"E:\Data_current\MRI\2021\McNally\McN202109_WT_170E_0930_LegT2\Analysis\nnls\170E_t2raw_masked.raw"
                 )

for f in filename_list:
    print("Processing " + f)
    full_vol=np.fromfile(f,dtype='double')
    full_vol_re = np.reshape(full_vol, (ii.nslices, ii.nechoes, ii.xdim, ii.ydim))
    t2_vol_arr = nnt2.fit_t2_nnls(full_vol_re, A_mat, num_t2s)
    
    # im_path2=os.path.join(wd,'McN202102_D2_642B_0209_DCE_cath_t2_nnls')
    
    low_t2_vals = np.sum(np.squeeze(t2_vol_arr[:,:,:,lo_range]),axis=3)
    
    low_t2_img = np.sum(np.squeeze(t2_vol_arr[:,:,:,lo_range]),axis=3)
    mid_t2_img = np.sum(np.squeeze(t2_vol_arr[:,:,:,mid_range]),axis=3)
    hi_t2_img = np.sum(np.squeeze(t2_vol_arr[:,:,:,hi_range]),axis=3)
    
    # save one copy of the fit data alongside the original raw data,
    # and a second copy in a pooled directory for further analysis
    savedir = os.path.dirname(f)
    filename = os.path.splitext(os.path.basename(f))[0]
    out_filename = filename + '_t2_vol_arr.npy'
    np.save(os.path.join(savedir,out_filename), t2_vol_arr)
    np.save(os.path.join(bulk_storage_dir,out_filename), t2_vol_arr)
    
    t2sum = np.sum(t2_vol_arr, axis=(0,1,2))

