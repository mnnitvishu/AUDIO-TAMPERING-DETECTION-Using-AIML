# Audio Tampering Detection Using AI/ML

## Overview
This project implements an AI/ML-based audio tampering detection system using MATLAB.

It detects suspicious modifications in audio recordings without requiring an original reference file.

## Features
- Audio preprocessing and normalization
- Audio segmentation into 1-second windows
- MFCC feature extraction
- Short-Time Fourier Transform (FFT-based spectral analysis)
- Energy and Zero Crossing Rate (ZCR)
- Spectral centroid and spectral flux
- Background noise consistency analysis
- Random Forest classification
- Tampering probability estimation
- Suspicious audio segment localization
- Confusion matrix and ROC curve evaluation

## Technologies Used
- MATLAB
- Digital Signal Processing (DSP)
- Machine Learning
- MFCC
- Random Forest Classifier

## Workflow
1. Load clean speech audio files.
2. Preprocess and normalize audio signals.
3. Generate synthetic tampered audio samples.
4. Extract forensic features from audio windows.
5. Train a Random Forest classifier.
6. Evaluate model performance.
7. Analyze a single suspect audio file.
8. Identify suspicious time intervals.

## Requirements
- MATLAB
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

## Usage
1. Add at least 20 clean `.wav` speech files to `clean_audio/`.
2. Set `RUN_TRAINING = true` for the first run.
3. Execute `main.m`.
4. The trained model is saved as `tamper_model.mat`.
5. Set `RUN_TRAINING = false` for subsequent testing.
6. Place the suspect audio file as `suspect.wav`.
7. Run the script to view the detection results.

## Disclaimer
This project uses synthetically generated tampering examples for training. Its performance on real-world manipulated recordings requires independent validation.

## Author
Vishu Sharma

B.Tech – Electronics and Communication Engineering
