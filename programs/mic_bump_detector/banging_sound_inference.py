# This program takes in an audio file, extracts features from it, then feeds those features into a pre-trained
# classifier (random forest) to predict whether the audio file has banging sounds in it or not.
# It expects the audio file be in the data/audio/ folder and the model to be pickled in the models/ folder
# To run: 
#    python -m mic_bump_detector_programs.banging_sound_inference --audiofile pedro_foley.wav

import argparse
import joblib
import argparse
import sys
import os
import numpy as np
import pandas as pd
from programs.general_purpose.feature_extraction import extract_features

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--audiofile', type=str, default=None)
  args = parser.parse_args()
  audiofile = args.audiofile
  filepath = os.path.join("data/audio/", audiofile)
  if audiofile is None:
    print("Please provide an audio file  using --audiofile")
    sys.exit(1)
  if not os.path.isfile(filepath):
    print(f"File {filepath} does not exist.")
    sys.exit(1)
  
  # Run the audio feature extraction program and retrieve the df that has the results
  df = extract_features(audiofile, None)

  # log transform the skewed variables
  skewed_variables = ['rms_avg', 'rms_max']
  for col in skewed_variables:
    df[f'{col}_log'] = np.log1p(df[col])

  # drop unecessary columns
  df = df.drop(columns=['filename', 'group', *skewed_variables])
  predictors = [
    'zcr_avg',
    'mfcc1_avg',
    'mfcc2_avg',
    'mfcc3_avg',
    'mfcc4_avg',
    'mfcc5_avg',
    'mfcc6_avg',
    'mfcc7_avg',
    'mfcc8_avg',
    'mfcc9_avg',
    'sc_avg',
    'rms_avg_log',
    'rms_max_log',
  ]
  df = df[[
    *predictors,
  ]]

  # Load the model
  model = joblib.load('models/RandomForestClassifier.pkl')

  # Run inference on new data
  y_pred = model.predict(df)
  print(f"Predicted class for {audiofile}: {y_pred[0]}")
  print('')
  if y_pred[0] == 0:
    print("Audio file likely does NOT contains banging sounds.")
  else:
    print("Audio file likely contains banging sounds.")


  # If you also want predicted probabilities
  y_pred_proba = model.predict_proba(df)
  print(f"Predicted probabilities for {audiofile}: {y_pred_proba[0]}")
