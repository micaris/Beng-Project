#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:34:32 2019

@author: michaelosinowo
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew, pearsonr
import seaborn as sns
import math
import tqdm
#import pytweening

# Reading the for dataset into dataframe
df = pd.read_csv("anscombes.csv")

# Extracting the four datasets
df1 = df[df['dataset'] == 'I']
x1, y1 = df1['x'].to_numpy(), df1['y'].to_numpy()
df2 = df[df['dataset'] == 'II']
x2, y2 = df2['x'].to_numpy(), df2['y'].to_numpy()
df3 = df[df['dataset'] == 'III']
x3, y3 = df3['x'].to_numpy(), df3['y'].to_numpy()
df4 = df[df['dataset'] == 'IV']
x4, y4 = df4['x'].to_numpy(), df4['y'].to_numpy()


#------------ Implementing the simulated annealing approach----------------#

# sample a new dataset from the joint distribution of anscombe's quartet -------------#

def gen_new_ds(low = 0, high= 12, size= (2, 11)):
    new_ds = np.random.uniform(low=low, high=high, size=size)
    return np.array([new_ds[0], new_ds[1]])

ref_stats = [9.000000000000000000e+00,
3.162277660168379523e+00,
7.500681818181818450e+00,
1.936536623931276013e+00,
8.163662996807959926e-01]


#function to perturb the datasets
def perturb(ds, initial, target_stat = ref_stats, shift_f=0.1, temp=0,):

    # take one row at random, and move one of the points a bit
    row = np.random.randint(0, len(df))
    

    new_ds = ds.copy
    
    # take one row at random, and move one of the points a bit
    row = np.random.randint(0, len(df))
    
    # this is the simulated annealing step, if "do_bad", then we are willing to
    # accept a new state which is worse than the current one
    do_bad = np.random.random_sample() < temp

    while True:
        xm = i_xm + np.random.randn() * shake
        ym = i_ym + np.random.randn() * shake


        if target == 'circle':
            # info for the circle
            cx = 54.26
            cy = 47.83
            r = 30
            
            dc1 = dist([df['x'][row], df['y'][row]], [cx, cy])
            dc2 = dist([xm, ym], [cx, cy])
            old_dist = abs(dc1 - r)
            new_dist = abs(dc2 - r)

        close_enough = (new_dist < old_dist or new_dist < allowed_dist or do_bad)
        within_bounds = ym > y_bounds[0] and ym < y_bounds[1] and xm > x_bounds[0] and xm < x_bounds[1]
        if close_enough and within_bounds:
            break

    # set the new data point, and return the set
    df['x'][row] = xm
    df['y'][row] = ym
    
    return df

#------error--------------#

def error(ds_stat, ref_stat):
    new_arr = np.subtract(ds_stat, ref_stat)
    return np.sum(np.square(np.abs(new_arr)))

#Main function 

def sim_ann(data, min_temp = 0, max_temp= 0.4, iters = 100):
    """
    This function takes in a dataset sampled from the joint
    distribution of the Anscombe's quartet and perturbs till
    same stat is acheived
    Args:
        data: dataset to be perturbed
        iters: the number of iterations to run for
        min_temp: T_o/starting temprature for the simulated anealing
        max_temp: T_f/maximun temprature '' '' '' ''
        shift: Factor by which the data point will be updated
    """
    temp = 0
    min_temp = 0
    max_temp = 0.4
    w , r = 0, 0
    for i in range(0, iters):
        do_bad = np.random.random_sample() < temp
        
        if do_bad:
            w += 1
        else:
            r += 1
            
        # updating the temprature i.e mimicing the cooling effect
        temp = (max_temp - min_temp) * ((iters - i) / iters) + min_temp
        
    print('Wrong Solutions accepted: {}'.format(w))
    print('Right Solutions accepted: {}'.format(r))
    
sim_ann()

#----------- Testing ----------------#
new_ds = gen_new_ds()
print(new_ds)
x , y = new_ds[0], new_ds[1]


print('------DATASET new_ds-------')

def get_sum_stats(x,y):
    print("X mean: ", x.mean())
    print("X SD: ", x.std())
    print("Y mean: ", y.mean())
    print("Y SD: ", y.std())
    print("Pearson correlation: ", pearsonr(x,y)[0])

    return [x.mean(), x.std(), y.mean(), y.std(), pearsonr(x,y)[0]]

xy = get_sum_stats(x,y)
 
xy1 = get_sum_stats(x1,y1)
xy2 = get_sum_stats(x2,y2)
xy3 = get_sum_stats(x3,y3)
xy4 = get_sum_stats(x4,y4)

err = error(xy, xy2)
err1 = error(xy1, xy2)
print(err, err1)

print(xy1)
x_ave = np.sum([xy1, xy2, xy3, xy4], axis = 0)
x_ave = x_ave/4
plt.scatter(x,y)
coef1 = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(coef1) 
plt.plot(x, poly1d_fn(x), color='black')
#----------- Testing ----------------#