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


global error_t

error_t = []

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


#------Implementing the simulated annealing approach----------------#


def gen_new_ds(low = 0, high= 12, size= (2, 11)):
    """
    This function produces a random dataset within the
    bounds of the Anscombe's quartet values.
    Args: 
        low: the lowest values that can be sampled
        high: the highest value that can be sampled
        size: the dimentions of the dataset to be generated
    """
    new_ds = np.random.uniform(low=low, high=high, size=size)
    
    return np.array([new_ds[0], new_ds[1]])

def get_sum_stats(x,y):
    """
    This function returns the summary statistics of 
    the given dataset.
    Args:
        x: the first row/column of the 2-D dataset
        y: second roe/column of the 2-D dataset
    
    """
    print("X mean: ", x.mean())
    print("X SD: ", x.std())
    print("Y mean: ", y.mean())
    print("Y SD: ", y.std())
    print("Pearson correlation: ", pearsonr(x,y)[0])

    return [x.mean(), x.std(), y.mean(), y.std(), pearsonr(x,y)[0]]

#-----average summary statistics for Anscombe's quartet-----#
ref_stats = [9.000000000000000000e+00,
3.162277660168379523e+00,
7.500681818181818450e+00,
1.936536623931276013e+00,
8.163662996807959926e-01]

#function to perturb the datasets
def perturb(ds, target_stat = ref_stats, shift= 0.4, 
                temp=0, 
                x_lim = [0,12],
                y_lim = [0,12]):
    """
    This function takes in a 2-D datset, and perturbs it till a
    conditions "close_enough" and "within_bounds" are met.
    
    """

    new_ds = ds.copy()
    x, y = new_ds[0], new_ds[1]
    
    # take one row at random, and move one of the points a bit
    row = np.random.randint(0, len(ds[0]))
    i_xm, i_ym = x[row], y[row]
    
    print('This is row : {}'.format(row))
    
    #get summary statistics and calculate error of current dataset
    init_stats = get_sum_stats(x,y)
    init_err = error(init_stats, ref_stats)
    
    #keep track of the current error
    error_t.append(init_err)
    
    # this is the simulated annealing step, if "do_bad", then we are willing to
    # accept a new state which is worse than the current one
    do_bad = np.random.random_sample() < temp

    while True:
        #perturb the values of the row chosen at random
        x[row] = i_xm + np.random.randn() * shift
        y[row] = i_ym + np.random.randn() * shift
        
        #calculate the new squared error
        curr_stats = get_sum_stats(x,y)
        curr_err = error(curr_stats, ref_stats)
        #determine if the error is better and values are within bounds
        close_enough = (curr_err < init_err  or do_bad)
        within_bounds = y[row] > y_lim[0] and y[row] < y_lim[1] and x[row] > x_lim[0] and x[row] < x_lim[1]
        
        if close_enough and within_bounds:
            break

    # set the new data point, and return the set
    new_ds[0]= x
    new_ds[1]= y
    
    return new_ds

#------error--------------#
def error(ds_stat, ref_stat):
    """
    This function returns the cummulative squared error 
    between the datasets statistics and a reference statistics
    Args:
        ds_stat: The statistics to be improved on
        ref_stats: The reference statistics
    """
    new_arr = np.subtract(ds_stat, ref_stat)
    return np.sum(np.square(np.abs(new_arr)))

def error_axis(mean, ref_mean):
    return abs((mean-ref_mean)**2)
     


def sim_ann(data, min_temp = 0, max_temp= 0.4, iters =200):
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
    new_ds = data.copy()
    
    for i in range(0, iters):
        print('on iteration: {}'.format(i))
        new_ds = perturb(new_ds, temp= temp)
        temp = (max_temp - min_temp) * ((iters - i) / iters) + min_temp
        
    return new_ds

#----------- Testing ----------------#
new_ds = gen_new_ds()
opt_ds = sim_ann(new_ds, iters=400)


print('------DATASET new_ds-------')


x,y = opt_ds[0], opt_ds[1]
x_i, y_i = new_ds[0] ,new_ds[1]

#print summary stats for the initial and final dataset
xy1 = get_sum_stats(x,y)
xy2 = get_sum_stats(x_i, y_i)

#plot error trend
plt.plot(error_t[200:])



#plot generated dataset
plt.scatter(opt_ds[0],opt_ds[1])
coef1 = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(coef1) 
plt.plot(x, poly1d_fn(x), color='black')
#----------- Testing ----------------#