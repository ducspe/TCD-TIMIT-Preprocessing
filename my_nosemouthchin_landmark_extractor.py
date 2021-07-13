import dlib
import cv2
import os
import time
from multiprocessing import Process
from collections import OrderedDict
import numpy as np
import skvideo.io

# The abbreviation nmc in this file means nose/mouth/chin region of interest
# dda means digital differential analyzer
base_dir = "original_timit_data/"

# Parameters:
epsilon = 1e-8  # for numerical stability
sampling_rate = 16000
visual_frame_rate_in = 29.970030
visual_frame_rate_out = 62.5
crf = 17

# Indices for dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def landmark_to_np(landmarks_in):
    # print("num_parts: ", landmarks_in.num_parts)  # this prints 68, i.e. total nr of landmarks
    landmark_arr = np.zeros(shape=(landmarks_in.num_parts, 2), dtype=int)
    for i in range(landmarks_in.num_parts):
        landmark_arr[i] = (landmarks_in.part(i).x, landmarks_in.part(i).y)

    return landmark_arr


def my_processing_per_cpu_process(speaker_sublist):
    print(f"Starting process with sublist {speaker_sublist}")

    for speaker in speaker_sublist:
        print(f"Processing speaker {speaker}")
        speaker_path = os.path.join(base_dir, speaker, "straightcam")

        speaker_mp4_files = sorted([file for file in os.listdir(speaker_path) if ".mp4" in file])
        speaker_audio_files = sorted([file for file in os.listdir(speaker_path) if ".wav" in file])

        for video_mp4_file, audio_wav_file in zip(speaker_mp4_files, speaker_audio_files):
            video_mp4_pa = os.path.join(speaker_path, video_mp4_file)

            original_video_array = skvideo.io.vread(video_mp4_pa)

            ### Create the dda file first
            os.makedirs(f'landmark_preprocessing_steps/{speaker}/',
                        exist_ok=True)  # speaker is set by main, not in this function
            dda_video = f'landmark_preprocessing_steps/{speaker}/{video_mp4_pa.split("/")[-1].replace(".mp4", "_dda.mp4")}'
            video_writer0 = skvideo.io.FFmpegWriter(dda_video,
                                                    inputdict={'-r': str(visual_frame_rate_in),
                                                               '-s': '{}x{}'.format(1920, 1080)},  # size of original videos
                                                    outputdict={'-filter:v': 'fps=fps={}'.format(
                                                        visual_frame_rate_out),
                                                        '-c:v': 'libx264',
                                                        '-s': '{}x{}'.format(224, 224),  # need to resize, otherwise it takes ages to finish processing this script
                                                        '-crf': str(crf),
                                                        '-preset': 'veryslow'}
                                                    )

            for dda_frame in original_video_array:
                video_writer0.writeFrame(dda_frame)

            video_writer0.close()

            ### End of dda creation

            dda_video_array = cv2.VideoCapture(dda_video)
            dda_video_length = int(dda_video_array.get(cv2.CAP_PROP_FRAME_COUNT))

            success, frame = dda_video_array.read()
            count_frame = 0

            nmc_landmarks_arr = np.zeros((dda_video_length, total_nr_of_landmarks, 2), dtype=int)
            # shape is (nr_frames, nr_landmarks_per_frame, 2 coordinates of a landmark)
            while success:
                detected_faces = detector(frame, 1)
                # 1 is the number of image pyramid layers to apply when upscaling the image

                # loop over all detected faces, which for TIMIT should be 1
                for (count_face, detected_face) in enumerate(detected_faces):
                    landmarks = predictor(frame, detected_face)
                    landmarks_arr = landmark_to_np(landmarks)

                    nmc_landmarks_arr[count_frame] = np.vstack((np.nan_to_num(landmarks_arr[nose_start_roi_index:nose_end_roi_index]), np.nan_to_num(landmarks_arr[mouth_start_roi_index:mouth_end_roi_index]), np.nan_to_num(landmarks_arr[chin_start_roi_index:chin_end_roi_index])))
                    # print("nmc arr shape: ", nmc_landmarks_arr.shape)  # prints 46, which is correct
                    # Plot to see correctness:
                    #
                    # image_copy = frame.copy()
                    # for (i, j) in landmarks_arr[start_roi_index:end_roi_index]:
                    #     cv2.circle(image_copy, (i, j), 1, (0, 0, 255), -1)  # cv2.circle(image, center_coordinates, radius, color, thickness)
                    #     # thickness=-1 fills the circle by the specified color, in this case red.
                    #
                    #     cv2.imshow("Image with landmark points", image_copy)
                    #     cv2.waitKey(0)  # display window infinitely until any keypress

                    # End of plotting code

                success, frame = dda_video_array.read()
                count_frame += 1

            # save the landmarks as npy file
            os.makedirs(f'dda_landmarks_dataout/{speaker}/', exist_ok=True)  # speaker is set by main, not in this function
            dda_landmarks_path = f'dda_landmarks_dataout/{speaker}/{video_mp4_pa.split("/")[-1].replace(".mp4", ".npy")}'
            print("Facial landmarks array before saving: ", nmc_landmarks_arr)
            np.save(dda_landmarks_path, nmc_landmarks_arr)


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Consider all the facial landmarks:
    nose_start_roi_index = FACIAL_LANDMARKS_68_IDS["nose"][0]
    nose_end_roi_index = FACIAL_LANDMARKS_68_IDS["nose"][1]

    mouth_start_roi_index = FACIAL_LANDMARKS_68_IDS["mouth"][0]
    mouth_end_roi_index = FACIAL_LANDMARKS_68_IDS["mouth"][1]

    chin_start_roi_index = FACIAL_LANDMARKS_68_IDS["jaw"][0]
    chin_end_roi_index = FACIAL_LANDMARKS_68_IDS["jaw"][1]

    total_nr_of_landmarks = nose_end_roi_index-nose_start_roi_index + mouth_end_roi_index-mouth_start_roi_index + chin_end_roi_index-chin_start_roi_index

    all_speaker_folders = sorted(os.listdir(base_dir))
    print("All speakers: ", all_speaker_folders)

    pool_list = [all_speaker_folders[5 * i:5 * i + 5] for i in
                 range(12)]  # each of the 12 CPU cores will process a maximum of 5 speaker folders.

    print("Pool list: ", pool_list)

    start_time = time.perf_counter()

    processes = []
    for split_list in pool_list:
        p = Process(target=my_processing_per_cpu_process,
                    args=(split_list,))
        p.start()
        processes.append(p)
        # p.join()  # if this join is uncommented than the multicore preprocessing is transformed to just single core sequential processing.

    for proc in processes:
        proc.join()

    end_time = time.perf_counter()

    print(f"Elapsed time for the ROI creation: {(end_time - start_time) / 3600} hours")
