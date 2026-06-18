import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import torchaudio
import random


def genSpoof_list2019(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_eval):
        for line in l_meta:
            #key = line.strip()
            #_,key,_,_,_,_,_,_,label,_ = line.strip().split()
            key,label= line.strip().split(" ")
            file_list.append(key)
        return file_list

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _,key,_,_,_,_,_,_,label,_ = line.strip().split()
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            #key= line.strip()
            parts = line.strip().split(' ')
            key = parts[1]
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             #_,key,_,_,_,_,_,_,label,_ = line.strip().split()
             parts = line.strip().split()
             key = parts[1]
             label = parts[-2]
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list




def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_ASVspoof2019_train(Dataset):
	def __init__(self,args,list_IDs, labels, base_dir,algo):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.algo=algo
            self.args=args
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
           return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            # audio_path = os.path.join(self.base_dir, 'flac', utt_id + '.flac')
            # X, fs = torchaudio.load(audio_path)
            # if fs != 16000:
            #     resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            #     X = resampler(X)
            X,fs = librosa.load(self.base_dir+utt_id+'.flac', sr=16000) 
            Y=process_Rawboost_feature(X,fs,self.args,self.algo)
            X_pad= pad(Y,self.cut)
            x_inp= Tensor(X_pad)
            target = self.labels[utt_id]
            
            return x_inp, target
            
            
class Dataset_ASVspoof2021_eval(Dataset):
	def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            # audio_path = os.path.join(self.base_dir, 'flac', utt_id + '.flac')
            # X, fs = torchaudio.load(audio_path)
            # if fs != 16000:
            #     resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            #     X = resampler(X)
            X, fs = librosa.load(self.base_dir+utt_id+'.wav', sr=16000)
            #X, fs = librosa.load(self.base_dir+utt_id, sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,utt_id
        
def center_crop_or_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """x: 1D tensor [T]; center-crop if long, symmetric zero-pad if short."""
    Tcur = x.numel()
    if Tcur == target_len:
        return x
    if Tcur > target_len:
        start = (Tcur - target_len) // 2
        return x[start:start + target_len]
    # pad
    pad_total = target_len - Tcur
    left = pad_total // 2
    right = pad_total - left
    return torch.nn.functional.pad(x, (left, right))

class Dataset_OOD_eval(Dataset):
    def __init__(self, list_IDs, base_dir, cut=64600, target_sr=16000):
        """
        list_IDs: list of filenames (with extension)
        base_dir: directory path ending with '/'
        cut: number of samples to return (4s @16k = 64000; you used 64600 so kept default)
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut
        self.target_sr = target_sr

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        path = os.path.join(self.base_dir, utt_id)

        # load
        wav, sr = torchaudio.load(path)  # [C, T], dtype=float32 in [-1,1] if PCM
        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # [1, T]
        else:
            wav = wav  # [1, T]

        # resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            wav = resampler(wav)

        # to 1D
        x = wav.squeeze(0)  # [T]

        # center 4s (or symmetric pad)
        x = center_crop_or_pad(x, self.cut)

        # numpy/torch Tensor as your model expects
        return x.float(), utt_id


class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
               '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        # audio_path = os.path.join(self.base_dir, 'flac', utt_id + '.flac')
        # X, fs = torchaudio.load(audio_path)
        # if fs != 16000:
        #     resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        #     X = resampler(X)
        X, fs = librosa.load(self.base_dir + utt_id, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id



        #--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature
