import librosa
from packages.processing.target import clean_speech_VAD
import os
import numpy as np
import cv2


# Define parameters:
# global_frame_rate = 29.970030  # frames per second
wlen_sec = 0.064  # window length in seconds
hop_percent = 0.25  # math.floor((1 / (wlen_sec * global_frame_rate)) * 1e4) / 1e4  # hop size as a percentage of the window length
win = 'hann'  # type of window function (to perform filtering in the time domain)
center = False  # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect'  # This argument is ignored if center = False
pad_at_end = True  # pad audio file at end to match same size after stft + istft

# Noise robust VAD
vad_quantile_fraction_begin = 0.5  # 0.93
vad_quantile_fraction_end = 0.55  # 0.99
vad_quantile_weight = 1.0  # 0.999
vad_threshold = 1.7

# Other parameters:
sampling_rate = 16000
dtype = 'complex64'
eps = 1e-8


def create_ground_truth_labels_from_path(audio_path):
    raw_clean_audio, Fs = librosa.load(audio_path, sr=sampling_rate)

    mask_labels = clean_speech_VAD(raw_clean_audio,
                           fs=sampling_rate,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    return mask_labels.T


def create_ground_truth_labels(raw_clean_audio):

    mask_labels = clean_speech_VAD(raw_clean_audio,
                           fs=sampling_rate,
                           wlen_sec=wlen_sec,
                           hop_percent=hop_percent,
                           center=center,
                           pad_mode=pad_mode,
                           pad_at_end=pad_at_end,
                           vad_threshold=vad_threshold)

    return mask_labels.T


def create_video_paths_list(base_path):
    video_paths_list = []
    speaker_folders = sorted([x for x in os.listdir(base_path)])
    for speaker in speaker_folders:
            speaker_path = os.path.join(base_path, speaker)
            speaker_mat_files = sorted([y for y in os.listdir(speaker_path)])

            for sentence_mat_file in speaker_mat_files:
                sentence_video_path = os.path.join(speaker_path, sentence_mat_file)
                video_paths_list.append(sentence_video_path)

    return video_paths_list


def create_audio_paths_list(base_path):
    audio_paths_list = []
    speaker_folders = sorted([x for x in os.listdir(base_path)])
    for speaker in speaker_folders:
            speaker_path = os.path.join(base_path, speaker, "straightcam")
            speaker_wav_files = sorted([y for y in os.listdir(speaker_path)])

            for sentence_wav_file in speaker_wav_files:
                sentence_audio_path = os.path.join(speaker_path, sentence_wav_file)
                audio_paths_list.append(sentence_audio_path)

    return audio_paths_list


def animate_npy_file(npy_path, scale_by_255=False):
    with open(npy_path, 'rb') as in_f:
        loaded_np_image = np.load(in_f)
        nr_frames = loaded_np_image.shape[0]
        print("Data in matrix form: ", loaded_np_image)
        for frame in range(nr_frames):
            if scale_by_255:
                cv2.imshow("Loaded image: ", 255 * loaded_np_image[frame, :, :])
            else:
                cv2.imshow("Loaded image: ", loaded_np_image[frame, :, :])

            cv2.waitKey(0)
