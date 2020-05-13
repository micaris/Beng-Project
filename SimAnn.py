#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 00:57:44 2020

@author: Osinowo Michael
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew, pearsonr, linregress
import seaborn as sns

class SimulatedAnnealing:
    
    def __init__(self, min_temp = 0, max_temp = 0.4, iters = 200, alpha= 0.4, x_lim = [4,14], y_lim =[4,13]):
        """
        This function takes in a dataset sampled from the joint
        distribution of the Anscombe's quartet and perturbs till
        same stat is acheived
        Args:
            data: dataset to be perturbed
            iters: the number of iterations to run for
            min_temp: T_o/starting temprature for the simulated anealing
            max_temp: T_f/maximun temprature '' '' '' ''
            alpha: Updating rate/ Learning rate
        """
        self._ERROR = []
        self.ref_stats = [9.0        , 3.31662479, 7.50090909, 2.03156814, 0.81642052,
       0.50009091, 3.00009091, 0.66654246]
        self.min_temp = 0
        self.max_temp = 0
        self.iters = iters
        self.alpha = alpha
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.op_ds = []
        self.data = self.gen_new_ds()

        
        
    def gen_new_ds(self, low = 4, high= 13, size= (2, 11)):
        """
        This function produces a random dataset within the
        bounds of the Anscombe's quartet values.
        Args: 
            low: the lowest values that can be sampled
            high: the highest value that can be sampled
            size: the dimentions of the dataset to be generated
        """
        self.new_ds = np.random.uniform(low=low, high=high, size=size)
        
        return np.array([self.new_ds[0], self.new_ds[1]]) 
    
    
    def get_sum_stats(self, x, y):
        """
        This function returns the summary statistics of 
        the given dataset.
        Args:
            x: the first row/column of the 2-D dataset
            y: second roe/column of the 2-D dataset
        
        """
        #print("X mean: ", x.mean())
        #print("X SD: ", x.std())
        ###print("Y mean: ", y.mean())
        ##print("Y SD: ", y.std())
        #print("Pearson correlation: ", pearsonr(x,y)[0])
        slope, intercept, r_value, _ , _ = linregress(x,y) 
   
        return np.array([x.mean(), x.std(), y.mean(), y.std(), pearsonr(x,y)[0], slope, intercept, r_value**2])

    def error(self ,ds_stat, ref_stat):
        """
        This function returns the cummulative squared error 
        between the datasets statistics and a reference statistics
        Args:
            ds_stat: The statistics to be improved on
            ref_stats: The reference statistics
        """
        new_arr = np.subtract(ds_stat, ref_stat)
        return np.sum(np.square(np.abs(new_arr)))
    
    def perturb(self, ds, temp=0):
        """
        This function takes in a 2-D datset, and perturbs it till a
        conditions "close_enough" and "within_bounds" are met.
        
        """
    
        new_ds = ds.copy()
        x, y = new_ds[0], new_ds[1]
        
        # take one row at random, and move one of the points a bit
        row = np.random.randint(0, len(ds[0]))
        i_xm, i_ym = x[row], y[row]
        
        #print('This is row : {}'.format(row))
        
        #get summary statistics and calculate error of current dataset
        init_stats = self.get_sum_stats(x,y)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        init_err = self.error(init_stats, self.ref_stats)
        
        #Log the current error
        self._ERROR.append(init_err)
        
        # this is the simulated annealing step, if "do_bad", then we are willing to
        # accept a new state which is worse than the current one
        do_bad = np.random.random_sample() < temp
    
        while True:
            #perturb the values of the row chosen at random
            x[row] = i_xm + np.random.randn() * self.alpha
            y[row] = i_ym + np.random.randn() * self.alpha
            
            #calculate the new squared error
            curr_stats = self.get_sum_stats(x,y)
            curr_err = self.error(curr_stats, self.ref_stats)
            #determine if the error is better and values are within bounds
            close_enough = (curr_err < init_err  or do_bad)
            within_bounds = (y[row] > self.y_lim[0]) and (y[row] < self.y_lim[1]) and (x[row] > self.x_lim[0]) and (x[row] < self.x_lim[1])
            
            if close_enough and within_bounds:
                break
    
        # set the new data point, and return the set
        new_ds[0]= x
        new_ds[1]= y
        
        return new_ds
    
    def run_anneal(self):
        
        
        temp = 0
        new_ds = self.data.copy()
        
        for i in range(0, self.iters):
            print('on iteration: {}'.format(i))
            new_ds = self.perturb(new_ds, temp= temp)
            temp = (self.max_temp - self.min_temp) * ((self.iters - i) / self.iters) + self.min_temp
            
        self.op_ds = new_ds

    def plot_error(self):
        plt.figure(100)
        plt.plot(np.log10(self._ERROR))
        plt.xlabel(" Iteration ")
        plt.xlabel(" Log_10(error) ")
        #plt.show()

    def plot_data(self):
        x,y = self.op_ds[0], self.op_ds[1]
        plt.figure(200)
        plt.scatter(x,y)
        coef1 = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef1) 
        plt.plot(x, poly1d_fn(x), color='black')
        #plt.show()  

    def run(self):
        self.run_anneal()
        self.plot_error()
        self.plot_data()

