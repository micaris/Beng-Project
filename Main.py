# -*- coding: utf-8 -*-
"""
Final Year Project
Author: Osinowo Michael
Supervisor: Roderich Gross

"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import tqdm
import pytweening

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

#Plotting datasets
fig, axis = plt.subplots(2,2)
#dataset 1
axis[0, 0].scatter(x1, y1, color='red')
coef1 = np.polyfit(x1,y1,1)
poly1d_fn = np.poly1d(coef1) 
axis[0, 0].plot(x1, poly1d_fn(x1), color='black')
axis[0, 0].set_title('Dataset 1')

#dataset 2
axis[0, 1].scatter(x2, y2, color='green')
coef2 = np.polyfit(x2,y2,1)
poly1d_fn = np.poly1d(coef2) 
axis[0, 1].plot(x2, poly1d_fn(x2), color='black')
axis[0, 1].set_title('Dataset 2')

#dataset 3
axis[1, 0].scatter(x3, y3, color='blue')
coef3 = np.polyfit(x3,y3,1)
poly1d_fn = np.poly1d(coef3) 
axis[1, 0].plot(x3, poly1d_fn(x3), color='black')
axis[1, 0].set_title('Dataset 3')

#dataset 4
axis[1, 1].scatter(x4, y4, color='black')
coef4 = np.polyfit(x4,y4,1)
poly1d_fn = np.poly1d(coef4) 
axis[1, 1].plot(x4, poly1d_fn(x4), color='black')
axis[1, 1].set_title('Dataset 4')

plt.subplots_adjust(wspace=0.5, hspace=0.5)

datasets = [df1, df2, df3, df4]

# function to print summary statistics
def print_stats(df_list):
    for i in range(0,len(df_list)):
        print('------DATASET {}-------'.format(i+1))
        print("N: ", len(df))
        print("X mean: ", df.x.mean())
        print("X SD: ", df.x.std())
        print("Y mean: ", df.y.mean())
        print("Y SD: ", df.y.std())
        print("Pearson correlation: ", df.corr().x.y)

# Printing the summary statistics of anscombe's quartet     
print_stats(datasets)



# ---------- recreating the Simulated Annealing approach ----------- #

#Reading datasets for test
data_cloud = pd.read_csv("random_cloud.csv")
data_dino = pd.read_csv("Datasaurus_data.csv")

def show_scatter_plot(data, linear_reg= True):
   
    sns.regplot( x='x', y='y', data=data, fit_reg=linear_reg,  line_kws={"linewidth": 3, "color": "black"})
    plt.tight_layout
    
show_scatter_plot(data_cloud, False)
show_scatter_plot(data_dino, False)

def is_kernel():
    """Detects if running in an IPython session
    """
    if 'IPython' not in sys.modules:
        # IPython hasn't been imported, definitely not
        return False
    from IPython import get_ipython
    # check for `kernel` attribute on the IPython instance
    return getattr(get_ipython(), 'kernel', None) is not None

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_values(df):

    xm = df.x.mean()
    ym = df.y.mean()
    xsd = df.x.std()
    ysd = df.y.std()
    pc = df.corr().x.y

    return [xm, ym, xsd, ysd, pc]

def is_error_still_ok(df1, df2, decimals=2):
    
    r1 = get_values(df1)
    r2 = get_values(df2)
    # check each of the error values to check if they are the same to the
    # correct number of decimals

    r1 = [math.floor(r * 10**decimals) for r in r1]
    r2 = [math.floor(r * 10**decimals) for r in r2]

    # we are good if r1 and r2 have the same numbers
    er = np.subtract(r1, r2)
    er = [abs(n) for n in er]

    return np.max(er) == 0

def perturb(df, initial, 
            target='circle',
            line_error=1.5,
            shake=0.1,
            allowed_dist=3,  # should be 2, just making it bigger for the sp example
            temp=0,
            x_bounds=[0, 100],
            y_bounds=[0, 100],
            custom_points=None):

    # take one row at random, and move one of the points a bit

    row = np.random.randint(0, len(df))
    i_xm = df['x'][row]
    i_ym = df['y'][row]


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





def show_scatter_and_results(df):

    """Creates a plot which shows both the plot and the statistical summary
    Args:
        df (pd.DataFrame):  The data set to plot
       labels (List[str]): The labels to use for
    """
    plt.figure(figsize=(12, 5))
    sns.regplot("x", y="y", data=df, ci=None, fit_reg=False,
               scatter_kws={"s": 50, "alpha": 0.7, "color": "black"})
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.tight_layout()

    res = get_values(df)
    fs = 30
    y_off = -5

    labels = ("X Mean", "Y Mean", "X SD", "Y SD", "Corr.")
    max_label_length = max([len(l) for l in labels])

    # If `max_abel_length = 10`, this string will be "{:<10}: {:0.9f}", then we
    # can pull the `.format` method for that string to reduce typing it
    # repeatedly
    formatter = '{{:<{pad}}}: {{:0.9f}}'.format(pad=max_label_length).format
    corr_formatter = '{{:<{pad}}}: {{:+.9f}}'.format(pad=max_label_length).format

    opts = dict(fontsize=fs, alpha=0.3)
    plt.text(110, y_off + 80, formatter(labels[0], res[0])[:-2], **opts)
    plt.text(110, y_off + 65, formatter(labels[1], res[1])[:-2], **opts)
    plt.text(110, y_off + 50, formatter(labels[2], res[2])[:-2], **opts)
    plt.text(110, y_off + 35, formatter(labels[3], res[3])[:-2], **opts)
    plt.text(110, y_off + 20, corr_formatter(labels[4], res[4], pad=max_label_length)[:-2], **opts)

    opts['alpha'] = 1
    plt.text(110, y_off + 80, formatter(labels[0], res[0])[:-7], **opts)
    plt.text(110, y_off + 65, formatter(labels[1], res[1])[:-7], **opts)
    plt.text(110, y_off + 50, formatter(labels[2], res[2])[:-7], **opts)
    plt.text(110, y_off + 35, formatter(labels[3], res[3])[:-7], **opts)
    plt.text(110, y_off + 20, corr_formatter(labels[4], res[4], pad=max_label_length)[:-7], **opts)
    plt.tight_layout(rect=[0, 0, 0.57, 1])




def save_scatter_and_results(df, iteration, dp=72):
    show_scatter_and_results(df)
    plt.savefig(str(iteration) + ".png", dpi=dp)
    plt.clf()
    plt.cla()
    plt.close()


def s_curve(v):
    return pytweening.easeInOutQuad(v)


def run_pattern(df, target,iters=100000, num_frames=100, decimals=2, shake=0.2, max_temp=0.4,
                min_temp=0, freeze_for=0, reset_counts=False, custom_points=False):

    """The main function, transforms one dataset into a target shape by
    perturbing it.

    Args:
        df: the initial dataset
        target: the shape we are aiming for
        iters: how many iterations to run the algorithm for
        num_frames: how many frames to save to disk (for animations)
        decimals: how many decimal points to keep fixed
        shake: the maximum movement for a single iteration
    """
    r_good = df.copy()
    # this is a list of frames that we will end up writing to file
    write_frames = [ int(round(x * iters)) for x in np.arange(0, 1, 1 / (num_frames - freeze_for))]


    extras = [iters] * freeze_for
    write_frames.extend(extras)

    # this gets us the nice progress bars in the notebook, but keeps it from crashing
    looper = tqdm.tnrange if is_kernel() else tqdm.trange 
    frame_count = 0

    # this is the main loop, were we run for many iterations to come up with the pattern
    for i in looper(iters + 1, leave=True, ascii=True, desc=target + " pattern"):
        
         t = (max_temp - min_temp) * s_curve(((iters - i) / iters)) + min_temp
        test_good = perturb(r_good.copy(), initial=df, target='circle', temp=t)
       
        # here we are checking that after the perturbation, that the statistics are still within the allowable bounds
        if is_error_still_ok(df, test_good, decimals):
            r_good = test_good

        # save this chart to the file
        for _ in range(write_frames.count(i)):
            save_scatter_and_results(
                r_good,
                '{}-image-{:05d}'.format(target, frame_count),
                150)
            # save_scatter(r_good, "{}-image-{:05d}".format(target, frame_count), 150)
            r_good.to_csv("{}-data-{:05d}.csv".format(target, frame_count))
            frame_count += 1


    return r_good



run_pattern(data_cloud, target, iters=200000, num_frames=5)



















