# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:19:57 2022

@author: lina3953
"""
import os
import numpy as np
from funcs import isInString
import scipy.io
from mat73 import loadmat
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, lfilter, sosfilt,iirfilter

#definision of filters

def notch_filter(data,f0,Q,fs):
    b, a = signal.iirnotch(f0, Q, fs)
    y = signal.lfilter(b,a,data)
    return y
def butter_bandpass_sos(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',output ='sos')
    return sos
    
def butter_bandstop_sos(lowcut, highcut, fs, order = 6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandstop',output ='sos')
    return sos

def butter_lowpass_sos(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, high, btype='low',output ='sos')
    return sos

def butter_highpass_sos(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    sos = butter(order, high, btype='high',output ='sos')
    return sos

def butter_bandpass_sos(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',output ='sos')
    return sos

def butter_bandpass_filter_sos(data, lowcut, highcut, fs, order=6):
    sos = butter_bandpass_sos(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y
    
def butter_bandstop_filter_sos(data, lowcut, highcut, fs, order=6):
    sos = butter_bandstop_sos(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def butter_lowpass_filter_sos(data, highcut, fs, order=2):
    sos = butter_lowpass_sos( highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def butter_highpass_filter_sos(data, highcut, fs, order=2):
    sos = butter_highpass_sos( highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y



class MassGen:
    def __init__(self, ind):
        self.folder = r'C:\Users\lina3953\OneDrive - Nexus365\Desktop\massdata\physionet.org\files\challenge-2018\1.0.0\training'
        self.all_files = os.listdir(self.folder)
        self.name_rec = self.all_files[ind]
        self.file_rec = os.path.join(self.folder, self.name_rec)    #folder containing the data
        for i in os.listdir(self.file_rec):
            if i.endswith('.mat') and isInString(i, 'arousal'):
                self.arousal_file = os.path.join(self.file_rec, i)
            if i.endswith('.mat') and not isInString(i, 'arousal'):
                self.data_file = os.path.join(self.file_rec, i)
        self.channels = {'F3':0, 'F4':1, 'C3':2, 'C4':3, 'O1':4, 'O2':5, 'E1':6}
        self.stages = ['nonrem3', 'nonrem2', 'nonrem1', 'rem', 'wake', 'undefined']

        
    def load_data(self):
        """
        this function loads the data :)
        """
        data = scipy.io.loadmat(self.data_file)
        data = data['val']
        return data
    
    def read_hypno(self):
        """
        reading the hypno file from the .mat file
        """
        arr = loadmat(self.arousal_file)
        hypno_lines = arr['data']['sleep_stages']
        return hypno_lines
        
    
    def load_hypnogram(self, resampling = False):
        """
        Returns one array the length of the data as a hypnogram
        """
        arr = loadmat(self.arousal_file)
        hypno_lines = arr['data']['sleep_stages']
        hypno_array = np.zeros((6, len(hypno_lines[self.stages[0]])))
        for num,i in enumerate(self.stages):
            hypno_array[num] = hypno_lines[i]
        hypnogram = np.zeros(len(hypno_lines[self.stages[0]]))
        for i in range(len(hypno_lines[self.stages[0]])):
            hypnogram[i] = np.argmax(hypno_array[:,i])
        if resampling:
            hypnogram = hypnogram[np.arange(0,len(hypnogram),2)]
        return hypnogram
    
    def extract_info_hypno(self):   #for testing
        """
        Function to return array where state changes happen
        """
        hypnogram = self.load_hypnogram()
        state_change = []
        data = self.load_data()[0]
        for num,i in enumerate(hypnogram[1:]):
            if hypnogram[num-1] != hypnogram[num]:
                state_change.append(num)
        state_change = np.insert(state_change,0,0)
        state_change = np.append(state_change, len(data))
        return state_change
    
    def extract_info_hypno_with_hypo(self, resampling = False):   #for testing
        """
        Function to return array where state changes happen with hypnogram
        """
        hypnogram = self.load_hypnogram(resampling)
        state_change = []
        data = self.load_data()[0]
        l = len(data)
        if resampling:
            l = int(len(data)/2)
        state = []
        for num,i in enumerate(hypnogram[1:]):
            if hypnogram[num-1] != hypnogram[num]:
                state.append(hypnogram[num])
                state_change.append(num)
        state_change = np.insert(state_change,0,0)
        state = np.insert(state,0,hypnogram[0])
        state_change = np.append(state_change, l)
        return state,state_change
    
    def return_all_epochs(self, resampling = True, fs_new = 100):
        """
        Function to return all 30s epochs with labels
        """
        state,state_change = self.extract_info_hypno_with_hypo(resampling)
        eeg = self.load_data()[0]
        epoch_length = 6000
        if resampling:
            eeg = resample(eeg, int(len(eeg)/2))
            epoch_length = 3000
        epochs = []
        epoch_label = []
        start_ind = []
        for i in range(len(state_change)-1):
            epochs_no = (state_change[i+1] - state_change[i])/epoch_length
            epochs_no = int(epochs_no)
            for j in range(epochs_no):
                epochs.append(eeg[int(state_change[i] + j*epoch_length):int(state_change[i] + (j+1)*epoch_length)])
                epoch_label.append(state[i])
                start_ind.append(int(state_change[i] + j*epoch_length))
        epochs = np.array(epochs)
        epoch_label = np.array(epoch_label)
        return epochs, epoch_label, start_ind
        
        
    def test(self):     #for testing
        data = self.load_data()
        t = np.linspace(0, len(data[0])/200, len(data[0]))
        hypnogram = self.load_hypnogram()
        changes = self.extract_info_hypno()
        ax1 = plt.subplot(211)
        plt.specgram(data[0], Fs = 200, cmap = 'jet', vmin = -30, vmax = 10)
        plt.subplot(212)
        plt.plot(t, hypnogram)
        # for i in changes:
        #     plt.axvline(t[i])
    
    def creat_epoch(self, epoch_len = 6000, num_epochs = 5, channel = 0):
        """
        This function returns a random set of epochs from one patient
        
        Parameters
        ----------
        epoch_len : length of epoch
            The default is 6000.
        num_epochs : number of epochs for the patient
            The default is 5.

        Returns
        -------
        epochs of chosen channel

        """
        epochs = []
        labels = []
        hypno_lines = self.read_hypno()
        hyp_arr = np.array( tuple(hypno_lines.values()) ).T
        stages = hypno_lines.keys()
        data = self.load_data()
        for i in range(num_epochs):
            cond = True
            while cond:
                ind = np.random.randint(0, len(data[0])-epoch_len)
                local_stage = np.argmax(hyp_arr[ind])
                local_label = hyp_arr[ind:ind+epoch_len, local_stage]
                if len(np.argwhere(local_label) == epoch_len):
                    epochs.append(data[channel, ind:ind+epoch_len])
                    labels.append(local_stage)
                    cond = False
        return epochs, labels
    
    # def possible_faster_function(self, epoch_len = 6000, num_epochs = 5, channel = 0):  #for testing
    #     epochs = []
    #     labels = []
    #     hypno_lines = self.read_hypno()
    #     hyp_arr = np.array( tuple(hypno_lines.values()) ).T
    #     stages = hypno_lines.keys()
    #     data = self.load_data()
    #     for i in range(num_epochs):
    #         ind = np.random.randint(0, len(data[0])-epoch_len)
    #         current_stage = np.argmax(hyp_arr[ind])
    #         cond = True
    #         k = 1
    #         while cond:
    #             next_stage = np.argmax(hyp_arr[ind+k])
    #             if next_stage != current_stage:
    #                 state_change = ind + k
    #                 cond = False
    #             k += 1
    #         ind = ind+k
    #         epochs.append(data[channel, ind:ind+epoch_len])
    #         labels.append(next_stage)
    #     return epochs, labels
    
    def creat_epoch_eog(self, epoch_len = 6000, num_epochs = 5, channel = 0, resampling = False, new_fs = 100):
        """
        This function returns a random set of epochs from one patient + the corresponding EOG epoch
        
        Parameters
        ----------
        epoch_len : length of epoch
            The default is 6000.
        num_epochs : number of epochs for the patient
            The default is 5.

        Returns
        -------
        epochs of chosen channel

        """
        epochs_eeg = []
        epochs_eog = []
        labels = []
        hypno_lines = self.read_hypno()
        hyp_arr = np.array( tuple(hypno_lines.values()) ).T
        stages = hypno_lines.keys()
        data = self.load_data()
        # if resampling:
        #     new_data = []
        #     for num,i in enumerate(data[0:7]):
        #         new_vals = resample(i, int(len(i)*new_fs/200))
        #         new_data.append(new_vals)
        #     new_data = np.array(new_data)
        #     data = new_data
        #     epoch_len = int(epoch_len*new_fs/200)
            
        for i in range(num_epochs):
            cond = True
            while cond:
                ind = np.random.randint(0, len(data[0])-epoch_len)
                local_stage = np.argmax(hyp_arr[ind])
                local_label = hyp_arr[ind:ind+epoch_len, local_stage]
                if len(np.argwhere(local_label) == epoch_len):
                    epochs_eeg.append(data[channel, ind:ind+epoch_len])
                    epochs_eog.append(data[6, ind:ind+epoch_len])
                    labels.append(local_stage)
                    cond = False
        if resampling:
            eeg = []
            eog = []
            for i,j in zip(epochs_eeg, epochs_eog):
                new_vals_eeg = resample(i, int(len(i)*new_fs/200))
                new_vals_eog = resample(j, int(len(j)*new_fs/200))
                eeg.append(new_vals_eeg)
                eog.append(new_vals_eog)
            epochs_eeg = np.array(eeg)
            epochs_eog = np.array(eog)
        return epochs_eeg, epochs_eog, labels
    
    
    
                    
    
    
                
        

