
# Program takes in an audio file and extracts features from it, and saves them as a row in data/output/output.csv
# for a new output.csv file, change header=True at the bottom
# To run: 
#    python -m analysis.feature_extraction --filename filename.wav

import argparse

import numpy as np
import librosa
import os
import sys
import pandas as pd

from utils.frequency_domain_utils import (
  short_time_fourier_transform,
)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--filename', type=str, default=None)
  args = parser.parse_args()
  filename = args.filename
  filepath = os.path.join("data/audio/", filename)
  if filename is None:
    print("Please provide a filename using --filename")
    sys.exit(1)
  if not os.path.isfile(filepath):
    print(f"File {filepath} does not exist.")
    sys.exit(1)
  
  # Load the audio file
  audio, sr = librosa.load(filepath, sr=16000)
  FRAME_LENGTH = 1024
  HOP_LENGTH = FRAME_LENGTH//4

  # Extract features
  rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
  zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
  stft = short_time_fourier_transform(audio, FRAME_LENGTH, HOP_LENGTH)
  mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, n_mfcc=13)
  delta_mfccs = librosa.feature.delta(mfccs)
  sc = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
  sban = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

  # Find averages, min, and max for each feature across whole file
  rms_avg = np.mean(rms)
  rms_min = np.min(rms)
  rms_max = np.max(rms)

  zcr_avg = np.mean(zcr)
  zcr_min = np.min(zcr)
  zcr_max = np.max(zcr)

  # simplify this code using a loop
  '''
  mfcc1_avg = np.mean(mfccs[0], axis=1)
  mfcc1_min = np.min(mfccs[0], axis=1)
  mfcc1_max = np.max(mfccs[0], axis=1)
  mfcc2_avg = np.mean(mfccs[1], axis=1)
  mfcc2_min = np.min(mfccs[1], axis=1)
  mfcc2_max = np.max(mfccs[1], axis=1)
  mfcc3_avg = np.mean(mfccs[2], axis=1)
  mfcc3_min = np.min(mfccs[2], axis=1)
  mfcc3_max = np.max(mfccs[2], axis=1)
  mfcc4_avg = np.mean(mfccs[3], axis=1)
  '''

mfcc_stats = {}
delta_mfcc_stats = {}
for i in range (mfccs.shape[0]):
  mfcc_stats[f'mfcc{i}_avg'] = np.mean(mfccs[i])
  mfcc_stats[f'mfcc{i}_min'] = np.min(mfccs[i])
  mfcc_stats[f'mfcc{i}_max'] = np.max(mfccs[i])
  delta_mfcc_stats[f'delta_mfcc{i}_avg'] = np.mean(delta_mfccs[i])
  delta_mfcc_stats[f'delta_mfcc{i}_min'] = np.min(delta_mfccs[i])
  delta_mfcc_stats[f'delta_mfcc{i}_max'] = np.max(delta_mfccs[i])

sc_avg = np.mean(sc)
sc_min = np.min(sc)
sc_max = np.max(sc)

print('sc_avg: ', sc_avg)
print('sc_min: ', sc_min)
print('sc_max: ', sc_max)

sban_avg = np.mean(sban)
sban_min = np.min(sban)
sban_max = np.max(sban)

audio_features = {
  'filename': filename,
  'rms_avg': rms_avg,
  'rms_min': rms_min,
  'rms_max': rms_max,
  'zcr_avg': zcr_avg,
  'zcr_min': zcr_min,
  'zcr_max': zcr_max,
  **mfcc_stats,
  **delta_mfcc_stats,
  'sc_avg': sc_avg,
  'sc_min': sc_min,
  'sc_max': sc_max,
  'sban_avg': sban_avg,
  'sban_min': sban_min,
  'sban_max': sban_max
}

# append results to csv file
columns=[
  'filename',
  'rms_avg','rms_min','rms_max',
  'zcr_avg','zcr_min','zcr_max',
  'mfcc0_avg','mfcc0_min','mfcc0_max',
  'mfcc1_avg','mfcc1_min','mfcc1_max',
  'mfcc2_avg','mfcc2_min','mfcc2_max',
  'mfcc3_avg','mfcc3_min','mfcc3_max',
  'mfcc4_avg','mfcc4_min','mfcc4_max',
  'mfcc5_avg','mfcc5_min','mfcc5_max',
  'mfcc6_avg','mfcc6_min','mfcc6_max',
  'mfcc7_avg','mfcc7_min','mfcc7_max',
  'mfcc8_avg','mfcc8_min','mfcc8_max',
  'mfcc9_avg','mfcc9_min','mfcc9_max',
  'mfcc10_avg','mfcc10_min','mfcc10_max',
  'mfcc11_avg','mfcc11_min','mfcc11_max',
  'mfcc12_avg','mfcc12_min','mfcc12_max',

  'delta_mfcc0_avg','delta_mfcc0_min','delta_mfcc0_max',
  'delta_mfcc1_avg','delta_mfcc1_min','delta_mfcc1_max',
  'delta_mfcc2_avg','delta_mfcc2_min','delta_mfcc2_max',
  'delta_mfcc3_avg','delta_mfcc3_min','delta_mfcc3_max',
  'delta_mfcc4_avg','delta_mfcc4_min','delta_mfcc4_max',
  'delta_mfcc5_avg','delta_mfcc5_min','delta_mfcc5_max',
  'delta_mfcc6_avg','delta_mfcc6_min','delta_mfcc6_max',
  'delta_mfcc7_avg','delta_mfcc7_min','delta_mfcc7_max',
  'delta_mfcc8_avg','delta_mfcc8_min','delta_mfcc8_max',
  'delta_mfcc9_avg','delta_mfcc9_min','delta_mfcc9_max',
  'delta_mfcc10_avg','delta_mfcc10_min','delta_mfcc10_max',
  'delta_mfcc11_avg','delta_mfcc11_min','delta_mfcc11_max',
  'delta_mfcc12_avg','delta_mfcc12_min','delta_mfcc12_max',

  'sc_avg','sc_min','sc_max',
  'sban_avg','sban_min','sban_max'
]
results = pd.DataFrame([audio_features], columns=columns)
results.to_csv("data/output/output.csv", mode="a", header=False, index=False)

