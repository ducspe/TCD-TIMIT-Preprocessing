### This repository contains 4 scripts to preprocess the TCD-TIMIT dataset
### The current multiprocessing setup works with 12 CPU cores, but it can be modified inside the main function.
### my_nosemouthchin_rgb_extractor.py will extract and process pixels within a rectangular bounding box around the nose, mouth and chin area
### my_nosemouthchin_landmark_extractor.py will extract only the landmarks, not all pixels of the nose, mouth and chin area. This representation is significantly lower in dimension as compared to RGB.
### my_fullface_rgb_extractor.py and my_fullface_landmark_extractor.py will extract rgb and landmarks respectively of the entire face, instead of focusing only on the nose, mouth and chin regions of interest as done by the scripts above.
### To verify qualitatively the results, utils.py and visualize_npy_samples.py are used. 
### During the execution of my_nosemouthchin_rgb_extractor.py, a separate folder is created that stores intermediate .mp4 representations of preprocessed data, with which synchronization between audio inferred labels and video frames can be verified.
### The scripts create "dataout" labeled folders with the extracted RGB, optical flow and landmark information in the form of .npy files for every speaker and every sentence uttered by that speaker.
### The folder "original_timit_data" contains a small subset of the original data to be processed. It's structure must be followed when working with the full TCD-TIMIT dataset.
