# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:39:27 2021

@author: alex
"""

import imageio, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stat


import nnls_T2_functions as nnt2
import image_info_constants as ii


#data_dir = r"C:\Users\alex\Box\DataCamp\NNLS_T2_fitting\TestData\NNLS_fits"
data_dir = r"E:\Data_current\MRI\2021\McNally\McN202109_165A-170E_nnls_fits"

subject_list = []
t2_vol_list = []
t2_avg_list = []
t2_log_list = []

for f in os.scandir(data_dir):
    try:
        t2_vol_arr = np.load(f)
    except:
        print('Unable to load ' + f)
    
    subject_list.append(f.name[0:4])
    t2_vol_list.append(t2_vol_arr)
    t2_avg = np.sum(t2_vol_arr,axis=(0,1,2))/(np.prod(t2_vol_arr.shape[0:3]))
    t2_avg_list.append(t2_avg)
    t2_log_list.append(np.log(t2_avg[1:-1]))

t2_df = pd.DataFrame(zip(subject_list, t2_vol_list, t2_avg_list, t2_log_list),columns=['id','nnls_t2_vol','roi_t2_dist','log_t2_dist'])
t2_df.set_index('id')

t2_df['t2_skew'] = t2_df['roi_t2_dist'].apply(stat.skew)
t2_df['t2_kurt'] = t2_df['roi_t2_dist'].apply(stat.kurtosis)

t2_df['log_t2_skew'] = t2_df['log_t2_dist'].apply(stat.skew)
t2_df['log_t2_kurt'] = t2_df['log_t2_dist'].apply(stat.kurtosis)

# This is dangerous - copied from fit_masked_t2_with_NNLS - needs to be saved and loaded somehow
# set up array of T2 values to use in dictionary
lo_max = 40
mid_max = 200
t2_max = 1000
num_t2s = 120
t2_arr = np.logspace(np.log10(10),np.log10(t2_max),num_t2s)

fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim((10,200))
for index, item in t2_df['log_t2_dist'].iteritems():
    plt.scatter(t2_arr[1:-1],item,label=t2_df['id'][index])
ax.legend()

ax2 = t2_df.plot.scatter(x='log_t2_skew', y='log_t2_kurt', c='DarkBlue')
