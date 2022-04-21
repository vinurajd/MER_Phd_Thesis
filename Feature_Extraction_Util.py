import librosa
import pandas as pd
import numpy as np

class Feature_Util:
    def __init__(self):
        ...
    # Function to read signal
    def readSignal(self, sound_file_path, sr=44100):
        signal_val = 0
        try:
            signal_val, sr = librosa.load(sound_file_path, sr=sr)
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

            if median_agg == "Y":
                median_val = np.median(sound_feature)
            if max_agg == "Y":
                max_val = np.max(sound_feature)
            # if min_agg == "Y":
            #     min_val = np.min(sound_feature)

            if sd_agg == "Y":
                sd_val = np.std(sound_feature)

            if IQR_agg == "Y":
                pval_75 , pval_25 = np.percentile(sound_feature,75) , np.percentile(sound_feature,25)
                IQR_val = pval_75 - pval_25

            ret_arr = {feature_name + "_mean": [round(mean_val,4)],
                       feature_name + "_median": [round(median_val, 4)],
                       feature_name + "_max": [round(max_val, 4)],
                       #feature_name + "_min": [round(min_val, 4)],
                       feature_name + "_sd": [round(sd_val, 4)],
                       feature_name + "_iqr": [round(IQR_val, 4)]}
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
            ret_val, msg_val = 1, "Successfully calculated RMS value."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting RMS value."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, rms_val

    def getSpectralCentroid(self, signal_val):
        spect_centroid = 0
        try:
            spect_centroid = librosa.feature.spectral_centroid(signal_val)
            ret_val, msg_val = 1, "Successfully calculated spectral centroid."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting spectral centroid."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, spect_centroid

    def getSpectralRolloff(self, signal_val):
        spect_rolloff = 0
        try:
            spect_rolloff = librosa.feature.spectral_rolloff(signal_val)
            ret_val, msg_val = 1, "Successfully calculated spectral rolloff."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting spectral rolloff."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, spect_rolloff

    def getSpectralBW(self, signal_val):
        spect_bw = 0
        try:
            spect_bw = librosa.feature.spectral_bandwidth(signal_val)
            ret_val, msg_val = 1, "Successfully calculated spectral bandwidth."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting spectral bandwidth."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, spect_bw

    def getSpectralContrast(self, signal_val):
        spect_contr = 0
        try:
            spect_contr = librosa.feature.spectral_contrast(signal_val)
            ret_val, msg_val = 1, "Successfully calculated spectral contrast."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting spectral contrast."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, spect_contr

    def getSpectralFlatness(self, signal_val):
        spect_flat = 0
        try:
            spect_flat = librosa.feature.spectral_flatness(signal_val)
            ret_val, msg_val = 1, "Successfully calculated spectral flatness."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting spectral flatness."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, spect_flat

    def getZCR(self, signal_val):
        zcr_val = 0
        try:
            zcr_val = librosa.feature.zero_crossing_rate(signal_val)
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

    def getMFCC_1(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[1]
            ret_val, msg_val = 1, "Successfully calculated MFCC 1."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_2(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[2]
            ret_val, msg_val = 1, "Successfully calculated MFCC 2."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_3(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[3]
            ret_val, msg_val = 1, "Successfully calculated MFCC 3."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_4(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[4]
            ret_val, msg_val = 1, "Successfully calculated MFCC 4."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_5(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[5]
            ret_val, msg_val = 1, "Successfully calculated MFCC 5."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_6(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[6]
            ret_val, msg_val = 1, "Successfully calculated MFCC 6."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_7(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[7]
            ret_val, msg_val = 1, "Successfully calculated MFCC 7."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_8(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[8]
            ret_val, msg_val = 1, "Successfully calculated MFCC 8."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_9(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[9]
            ret_val, msg_val = 1, "Successfully calculated MFCC 9."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_10(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[10]
            ret_val, msg_val = 1, "Successfully calculated MFCC 10."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_11(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[11]
            ret_val, msg_val = 1, "Successfully calculated MFCC 11."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_12(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[12]
            ret_val, msg_val = 1, "Successfully calculated MFCC 12."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

    def getMFCC_13(self, signal_val, sr):
        power_list = []
        try:
            mfcc_list = librosa.feature.mfcc(signal_val, sr, n_mfcc=15)[13]
            ret_val, msg_val = 1, "Successfully calculated MFCC 13."
        except Exception as e:
            ret_val = 0
            msg_val = "Error getting MFCC."+str(e.__class__)+"."+str(e)
        return ret_val, msg_val, mfcc_list

#utils_obj = Feature_Util()
#test_path = "D:/PhD Program/Final Research/DATASETS - Final Paper/DS - 1 MER_audio_taffc_dataset/Q1/MT0000040632.mp3"
# test_path = "D:/PhD Program/Final Research/DATASETS - Final Paper/DS - 1 MER_audio_taffc_dataset//Q2/MT0000971834.mp3"
# ret_val, msg_val, signal_val, sr = utils_obj.readSignal(test_path)
# #power_val = librosa.feature.rms(signal_val)
# print(zcr_val)
# ret_arr, msg_str = utils_obj.calcAggregations(zcr_val[0], feature_name="zcr")
# print(ret_arr)