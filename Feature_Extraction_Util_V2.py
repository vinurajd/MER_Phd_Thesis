import librosa
import pandas as pd
import numpy as np
from pydub import AudioSegment, silence
from scipy.signal import hilbert

class Feature_Util:
    def __init__(self):
        ...
    # Function to read signal
    def readSignal(self, sound_file_path, sr=44100, hilbert_transform='N'):
        signal_val = 0
        try:
            signal_val, sr = librosa.load(sound_file_path, sr=sr)
            if hilbert_transform =='Y':
                signal_val = np.abs(hilbert(signal_val))

            ret_val, msg_val = 1, "Successfully read sound file from path :" + str(sound_file_path)

        except Exception as e:
            ret_val, msg_val = 0, "Error reading file from path: "+str(sound_file_path)+"."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, signal_val, sr
    # 1. Function to calculate fundamental frequency (f0) using YIN algorithm (through librosa)

    def calcAggregations(self, sound_feature, feature_name, mean_agg = "Y", median_agg = "Y", max_agg = "Y", min_agg = "Y", IQR_agg = "Y", sd_agg ="Y"):
        feature_name = str(feature_name).lower()
        ret_arr = {}
        mean_val, median_val, max_val, min_val, sd_val, IQR_val = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        try:
            if mean_agg == "Y":
                mean_val = np.mean(sound_feature)
                temp_dict = {feature_name + "_mean": [round(mean_val, 4)]}
                ret_arr.update(temp_dict)

            if median_agg == "Y":
                median_val = np.median(sound_feature)
                temp_dict = {feature_name + "_median": [round(median_val, 4)]}
                ret_arr.update(temp_dict)

            if max_agg == "Y":
                max_val = np.max(sound_feature)
                temp_dict = {feature_name + "_max": [round(max_val, 4)]}
                ret_arr.update(temp_dict)

            if min_agg == "Y":
                min_val_sf = [sf_val for sf_val in sound_feature if sf_val > 0.0]
                min_val = np.min(min_val_sf)
                temp_dict = {feature_name + "_min": [round(min_val, 4)]}
                ret_arr.update(temp_dict)

            if sd_agg == "Y":
                sd_val = np.std(sound_feature)
                temp_dict = {feature_name + "_sd": [round(sd_val, 4)]}
                ret_arr.update(temp_dict)

            if IQR_agg == "Y":
                pval_75 , pval_25 = np.percentile(sound_feature,75) , np.percentile(sound_feature,25)
                IQR_val = pval_75 - pval_25
                temp_dict = {feature_name + "_iqr": [round(IQR_val, 4)]}
                ret_arr.update(temp_dict)
            #print(ret_arr)
            # ret_arr = {feature_name + "_mean": [round(mean_val,4)],
            #            feature_name + "_median": [round(median_val, 4)],
            #            feature_name + "_max": [round(max_val, 4)],
            #            feature_name + "_min": [round(min_val, 4)],
            #            feature_name + "_sd": [round(sd_val, 4)],
            #            feature_name + "_iqr": [round(IQR_val, 4)]}
            msg_str = "Successfully calculated aggregation for sound feature: " + str(feature_name)
        except Exception as e:
            msg_str = "Error calculating aggregation for sound feature: " + str(feature_name)+"."+str(e.__class__)+"."+str(e)
        return ret_arr, msg_str

    def getFZero(self, signal_val, fmin_val_note ="C3", fmax_val_note ="C6"):
        fund_freq = 0
        try:
            fund_freq = librosa.yin(signal_val,
                                    fmin=librosa.note_to_hz(fmin_val_note),
                                    fmax=librosa.note_to_hz(fmax_val_note))

            ret_val, msg_val = 1, "Successfully calculated F0."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting fundamental frequency."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, fund_freq

    def getRMS(self, signal_val):
        rms_val = 0
        try:
            rms_val = librosa.feature.rms(signal_val)
            rms_val = rms_val[0]
            ret_val, msg_val = 1, "Successfully calculated RMS value."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting RMS value."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, rms_val

    def getSpectralCentroid(self, signal_val):
        spect_centroid = 0
        try:
            spect_centroid = librosa.feature.spectral_centroid(signal_val)
            spect_centroid = spect_centroid[0]
            ret_val, msg_val = 1, "Successfully calculated spectral centroid."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting spectral centroid."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, spect_centroid

    def getSpectralRolloff(self, signal_val):
        spect_rolloff = 0
        try:
            spect_rolloff = librosa.feature.spectral_rolloff(signal_val)
            spect_rolloff = spect_rolloff[0]
            ret_val, msg_val = 1, "Successfully calculated spectral rolloff."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting spectral rolloff."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, spect_rolloff

    def getSpectralBW(self, signal_val):
        spect_bw = 0
        try:
            spect_bw = librosa.feature.spectral_bandwidth(signal_val)
            spect_bw = spect_bw[0]
            ret_val, msg_val = 1, "Successfully calculated spectral bandwidth."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting spectral bandwidth."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, spect_bw

    def getSpectralContrast(self, signal_val):
        spect_contr = 0
        try:
            spect_contr = librosa.feature.spectral_contrast(signal_val)
            spect_contr = spect_contr[0]
            ret_val, msg_val = 1, "Successfully calculated spectral contrast."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting spectral contrast."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, spect_contr

    def getSpectralFlatness(self, signal_val):
        spect_flat = 0
        try:
            spect_flat = librosa.feature.spectral_flatness(signal_val)
            spect_flat = spect_flat[0]
            ret_val, msg_val = 1, "Successfully calculated spectral flatness."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting spectral flatness."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, spect_flat

    def getZCR(self, signal_val):
        zcr_val = 0
        try:
            zcr_val = librosa.feature.zero_crossing_rate(signal_val)
            zcr_val = zcr_val[0]
            ret_val, msg_val = 1, "Successfully calculated zero crossing rate."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting zero crossing rate."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, zcr_val

    def getTempo(self, signal_val, sr):
        tempo_val = [0]
        try:
            onset_env = librosa.onset.onset_strength(signal_val)
            tempo_val = librosa.beat.tempo(onset_envelope= onset_env, sr=sr)
            ret_val, msg_val = 1, "Successfully calculated tempo."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting beat."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, round(tempo_val[0],4)

    def getPLP(self, signal_val, sr):
        plp_val = 0
        try:
            onset_env = librosa.onset.onset_strength(signal_val)
            plp_val = librosa.beat.plp(onset_envelope= onset_env, sr=sr)
            ret_val, msg_val = 1, "Successfully calculated plp."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting beat."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, plp_val

    def getPower(self, signal_val, sr):
        power_list = []
        try:
            S = librosa.feature.melspectrogram(signal_val, n_mels=128, sr=sr)
            for idx, power_val in enumerate(librosa.power_to_db(S)):
                power_list.append(max(power_val))

            ret_val, msg_val = 1, "Successfully calculated power."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting power."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, power_list

    def getMFCC(self, signal_val, sr, pos_val):
        mfcc_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=40)[pos_val]
            ret_val, msg_val = 1, "Successfully calculated MFCC " + str(pos_val)
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMel(self, signal_val, sr, pos_val, cutoff_val = 0.0):
        mel_list = []
        try:
            mel_list_all = librosa.feature.melspectrogram(signal_val, sr, n_mels=128)[pos_val]
            mel_list = [mel_list_val for mel_list_val in mel_list_all if mel_list_val > cutoff_val]
            ret_val, msg_val = 1, "Successfully calculated Mel " + str(pos_val)
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting Mel Freq."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mel_list
    def getLoudness(self, file_path):
        loudness = 0
        try:
            song_val = AudioSegment.from_mp3(file_path)
            ret_val, msg_val = 1, "Successfully calculated loudness "
            loudness = song_val.dBFS
        except Exception as e:
            ret_val=0
            msg_val = "Error getting Loudness from pydub."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, loudness
    def getChromagram(self,signal_val, sr):
        # This function would be used later
        chromagram = []
        try:
            chromagram = librosa.feature.chroma_stft(y=signal_val, sr=sr, hop_length=512)
            print(chromagram)
            print(chromagram[0])
            ret_val, msg_val = 1, "Successfully calculated Chromagram "
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting chromagram."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, chromagram



