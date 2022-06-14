#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:27:54 2022

@author: Quinn Cabooter
"""

#%%
"The preprocessing is done, following is the analyses of preprocessed data"

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import glob
from scipy import stats
from mne.stats import permutation_t_test
import pandas as pd

#%%
###############################################################################
#Import data
###############################################################################
#get channel information
# define your working directory: 


raw = mne.read_epochs('/Users/Quinn/Documents/MATLAB/MNE files/P2__clean_data.fif')#adapt path to match yours
raw.info


chan_order = raw.ch_names

################

# define your working directory (change this to yours)
working_dir =  "/Users/Quinn/Documents/MATLAB/MNE files/"
os.chdir(working_dir)

# find unique files ending with .fif
files = glob.glob('/Users/Quinn/Documents/MATLAB/MNE files/*clean_data.fif')
# take a look at the filenames and make sure they make sense
print(files)


# Next, we are going to loop over all these datafiles, 
# and compute averages at each timepoint, seperately for all conditions.
# In order to save these, we first create an empty "container"
epochs_all = dict(famous = [], scrambled = [], unfamiliar = [])

# Loop over datasets and add averaged data, per participant, to the containers
for sub in files: 
    # Load each dataset
    epochs = mne.read_epochs(sub, preload=True)
    epochs.apply_baseline(baseline=(-0.2, 0))
    
    # inspect ERPs to each conddition in each subject
    fam_epoch = epochs['famous'].average()
    scram_epoch = epochs['scrambled'].average()
    unfam_epoch = epochs['unfamiliar'].average()
    
    #uncomment to see each ERP for each pp separately, add unfamiliar = unfam_epoch to dictionary to add unfamiliar ERP
    #mne.viz.plot_compare_evokeds(dict(famous=fam_epoch, scrambled =scram_epoch),picks=epochs.ch_names.index('EEG065'), colors=('Blue','Red','Green'))
    # Now, we can average per condition
    for condition in ['famous', 'scrambled', 'unfamiliar']:
        temp_data1 = epochs[condition].get_data() 
        temp_data2 = np.mean(temp_data1,axis=0)
        
        epochs_all[condition].append(temp_data2)    

#create a variable N_datasets which indicates the number of participants  
N_datasets = len(files)
print(N_datasets)


times = epochs.times 
plt.plot(times); plt.ylabel('Time in the trial');plt.xlabel('Datapoints in the trial')

# Next, we want to look at average amplitudes at a specific electrode.
# In order to do so, we will first create empty 2D arrays [participants, times]
famous = np.empty([N_datasets,epochs_all['famous'][0].shape[1]]) #create empty array
scrambled = np.empty([N_datasets,epochs_all['scrambled'][0].shape[1]])
unfamiliar = np.empty([N_datasets,epochs_all['unfamiliar'][0].shape[1]])

#In order to avoid 'implicit multiple comparisons', 
# we here make an priori choice about the electrode
# The code below then loops over participants and adds average amplitude 
# at each timepoint at this specific electrode to your empty arrays.

whichElectrode = epochs.ch_names.index('EEG065')
for sub in range(0,N_datasets):
    famous[sub] = epochs_all['famous'][sub][whichElectrode]
    scrambled[sub] = epochs_all['scrambled'][sub][whichElectrode ]
    unfamiliar[sub] = epochs_all['unfamiliar'][sub][whichElectrode ]

# PLotting! Now, we can make an even-related plot, across participants, with seperate lines for conditions
# as well as variance
#to add unfamiliar to the graph uncomment the last two lines
plt.figure()
fam, = plt.plot(times,np.mean(famous,axis=0),color='blue',label='famous')
plt.fill_between(times,np.mean(famous,axis=0)+(np.std(famous,axis=0)/np.sqrt(N_datasets)),np.mean(famous,axis=0)-(np.std(famous,axis=0)/np.sqrt(N_datasets)),alpha=.2,color='blue')
scram, = plt.plot(times,np.mean(scrambled,axis=0),color='red',label='scrambled')
plt.fill_between(times,np.mean(scrambled,axis=0)+(np.std(scrambled,axis=0)/np.sqrt(N_datasets)),np.mean(scrambled,axis=0)-(np.std(scrambled,axis=0)/np.sqrt(N_datasets)),alpha=.2,color='red')
#unfam, = plt.plot(times,np.mean(unfamiliar,axis=0),color='green',label='unfamiliar')
#plt.fill_between(times,np.mean(unfamiliar,axis=0)+(np.std(unfamiliar,axis=0)/np.sqrt(N_datasets)),np.mean(unfamiliar,axis=0)-(np.std(unfamiliar,axis=0)/np.sqrt(N_datasets)),alpha=.2,color='green')


plt.legend();plt.title('Grand-Average ERP at site {}'.format(chan_order[whichElectrode]));plt.hlines(0,min(times),max(times),linestyles='dashed',color='grey');plt.vlines(0,min(scrambled.get_data(0)[1]),max(famous.get_data(0)[1]),linestyles='dashed',color='grey')
plt.xlabel('Time');plt.ylabel('µV');plt.text(0,max(famous.get_data(0)[1]),chan_order[whichElectrode])



meandifference=np.mean(famous,axis=0)-np.mean(scrambled,axis=0)
stddifference=np.std(famous-scrambled,axis=0)
difference, = plt.plot(times,meandifference,color='orange',label='difference')
plt.legend()


#plot all ERPs for participants individually
pick_color = [0.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
reds = plt.cm.get_cmap('Reds')
blues = plt.cm.get_cmap('Blues')
greens = plt.cm.get_cmap('Greens')

plt.figure()
for pp in range(0,N_datasets):
    blue1 = reds(pick_color[pp])
    fam, = plt.plot(times,famous[pp],label='famous', color = blue1)

    red1 = blues(pick_color[pp])
    scram, = plt.plot(times,scrambled[pp],label='scrambled', color = red1)


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='Blue'), Line2D([0], [0], color='Red')]

plt.legend(custom_lines, ['Famous', 'Scrambled'])


# Time for stats

#A-priori time window
P2_onset = 0.16
P2_offset =.18

#Then, we'll create a temporal mask, and average the amplitude within the specified timewindow
temporal_mask = np.logical_and(P2_onset <= times, times <= P2_offset)

famous_mean = np.mean(famous[:,temporal_mask],axis=1)
scrambled_mean = np.mean(scrambled[:,temporal_mask],axis=1)
meandifference = famous_mean - scrambled_mean

#Perform the test 
stats.ttest_1samp(meandifference,0, alternative ='less')

#For each participant separate
stats.ttest_1samp(famous_mean,scrambled_mean, alternative ='less')





# Permutation test
n_permutations = 5000
T0, p_values, H0 = permutation_t_test( famous - scrambled, n_permutations)

# We will add the significant time window as a horizontal to the plot created above.
# To do so, we turn all n.s. timeperiods into NaN, and all other to a specific value that will be plotted
# The line that is added to the plot shows you when there is a significant difference
p_values[p_values>.05] = np.NaN;p_values = np.ma.masked_where(np.isnan(p_values), p_values)
p_values[p_values<.05] = -.000003

# If the figure created above is still open, the code below adds the significant areas on the ERP plot
plt.plot(times,p_values,'k-', label = 'significant difference') 
plt.legend()