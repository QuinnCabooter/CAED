#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:02:41 2022

@author: Quinn Cabooter
"""

#%%%%%%%%%%%%%%%%%%
###############################################################################
#                            """ housekeeping"""
###############################################################################

#Import the necessary Python modules:
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
#%%
###############################################################################
#                           """import data"""
###############################################################################

# define your working directory: 
working_dir = "/Users/Quinn/Documents/MATLAB/Data All/" #change this to match yours
MNE_dir = "/Users/Quinn/Documents/MATLAB/MNE files/"

raw = mne.io.read_raw_eeglab(working_dir + 'sub-014_task-FaceRecognition_eeg.set',preload = True)
#raw.plot()

raw.drop_channels(['EEG061', 'EEG062', 'EEG063', 'EEG064', 'EEG071', 'EEG072', 'EEG073', 'EEG074'])
raw.info

#%%
###############################################################################
#                            """filter the data"""
###############################################################################

filt_raw = raw.copy() #create a copy of the raw data and name it "filt_raw"
filt_raw.load_data().filter(l_freq=1, h_freq=40) #apply the bandpass filter

#filt_raw.plot_psd() #frequency domain -> this is the power spectrum of the filtered EEG data
#%%
###############################################################################
#               """remove and interpolate bad channels"""
###############################################################################
# Mark bad channels
filt_raw.plot(n_channels =70)  
# Check list
print(filt_raw.info['bads'])


# Interpolate the bad channels
interp_filt_raw = filt_raw.copy()
interp_filt_raw.load_data().interpolate_bads(reset_bads=False)

#%%
###############################################################################
#                           """epoch data"""
###############################################################################
# store events from the RAW dataset
Event_info = mne.events_from_annotations(raw)

events = Event_info[0]
event_dict = Event_info[1]

#merge famous, scrambled and unfamiliar events into 1
events = mne.merge_events(events, [2, 3, 4], 2)
events = mne.merge_events(events, [7, 8, 9], 3)
events = mne.merge_events(events, [10, 11, 12], 4)

event_dict = {'boundary': 1,'famous': 2,'scrambled': 3,'unfamiliar': 4, 'left_nonsym': 5,'right_sym': 6}

print(np.unique(events[:, -1]))

# create epochs
epochs = mne.Epochs(interp_filt_raw, events, event_id = event_dict, tmin=-0.2, tmax=1,  
                     proj=False, baseline=(None, 0),  
                     preload=True, reject=None)  

#%%
###############################################################################
#                   """re-reference to the average"""
###############################################################################

epochs.set_eeg_reference().apply_proj().average()


evoked_pre_average = epochs.average()
evoked_pre_average.plot()

#%%
###############################################################################
#                   """visual artifact rejection"""
###############################################################################
# Mark bad trials
epochs.plot(n_epochs = 35, n_channels = 70)



# - Adjust the scale to a comfortable level (HELP to see keyboard controls)
# - Make sure you see ALL channels (Page Up key)
# - Then start at the beginning and click on epochs you think should be rejected
# - Don't reject trials with eye blinks or eye movements; we will get these with ICA

# Save the clean data set
epochs.save(MNE_dir + 'P14_epoched.fif', overwrite=True)

#%%

###############################################################################
#               """Independent Components Analysis"""
###############################################################################

ncomp =  len(epochs.info['ch_names']) - len(epochs.info['bads']) - 1
print(ncomp) #This should be equal to the original number of channels - number of interpolated channels - 1

# create ICA object with desired parameters
ica = mne.preprocessing.ICA(n_components = ncomp)

# do ICA decomposition
ica.fit(epochs) 

# Plot the components 
#ica.plot_sources(epochs)

# Their topography
ica.plot_components()

# Plot the properties of a single component (e.g. to check its frequency profile)
ica.plot_properties(epochs, picks=[3])

# Decide which component(s) to reject ("project out of the data" / "zero out")

#%%

###############################################################################
#               """Preprocess the raw data again"""
###############################################################################

# Open the raw data file 
raw = mne.io.read_raw_eeglab(working_dir + 'sub-014_task-FaceRecognition_eeg.set')


raw.drop_channels(['EEG061', 'EEG062', 'EEG063', 'EEG064', 'EEG071', 'EEG072', 'EEG073', 'EEG074'])
raw.info

# filter the data with a bandpass filter from 0.1 Hz and 40 Hz
refilt_raw = raw.copy() 
refilt_raw = raw.filter(l_freq=0.1, h_freq=40)

# Copy the list of bad channels from before to the new data set 
refilt_raw.info['bads'] = epochs.info['bads'] 

# Remove and interpolate the bad channels
#There was no need for interpolating bad channels, however due to issues with the dataset this did not work when trying 
#to do it for a random electrode.
interp_refilt_raw = refilt_raw.copy()
interp_refilt_raw = interp_refilt_raw.load_data().interpolate_bads(reset_bads=True)


# Epoching the data
# store events from the RAW dataset
Event_info = mne.events_from_annotations(raw)

events = Event_info[0]
event_dict = Event_info[1]

#merge famous, scrambled and unfamiliar events into 1
events = mne.merge_events(events, [2, 3, 4], 2)
events = mne.merge_events(events, [7, 8, 9], 3)
events = mne.merge_events(events, [10, 11, 12], 4)

event_dict = {'boundary': 1,'famous': 2,'scrambled': 3,'unfamiliar': 4, 'left_nonsym': 5,'right_sym': 6}

epochs = mne.Epochs(interp_refilt_raw, events, event_id = event_dict, tmin=-0.2, tmax=1,  
                     proj=False, baseline=(-0.2, -0.05),  
                     preload=True, reject=None)  

# Apply an average reference 
epochs.set_eeg_reference().apply_proj().average()

# Now we take the ICA weights calculated previously on the cleaned and heavily filtered data
# And apply it to the original, refiltered data
ica_epochs = epochs.copy() #create a copy of the raw data and name it ica_raw
ica_epochs = ica.apply(ica_epochs) #Apply the weights of the ICA to the copy of the raw data
#if you check your output, you should see that the selected components are zeroed out

# Plot the data with and without ICA
epochs.plot(n_epochs = 5, n_channels = 10)
ica_epochs.plot(n_epochs = 5, n_channels = 10)


#%%

###############################################################################
#                  """second visual artifact rejection"""
###############################################################################
# Now that we have corrected artifacts with ICA, we still need to remove bad trials.


# Plot the epochs to do visual trial rejection.
ica_epochs.plot(n_epochs = 35, n_channels = 70)

#Save clean data
ica_epochs.save(MNE_dir + 'P14__clean_data.fif', overwrite=True)


###############################################################################
#                  """create some ERP's"""
###############################################################################

#Loop over conditions and plotting
for cond in ['famous', 'scrambled', 'unfamiliar']:
    ica_epochs[cond].average().plot_joint()
 
 
#Create an average over all epochs, separately for the different conditions             
famous = ica_epochs['famous'].average()
scrambled = ica_epochs['scrambled'].average()
unfamiliar = ica_epochs['unfamiliar'].average()


#famous.plot_image()
#scrambled.plot_image()
#unfamiliar.plot_image()

 

#Compare conditions in a ERP plot
mne.viz.plot_compare_evokeds(dict(famous=famous, crambled = scrambled, unfamiliar = unfamiliar),
                             picks = [59], show_sensors=True)
 
 
#Plot topoplots
famous.plot_topo()
scrambled.plot_topo()
unfamiliar.plot_topo()




