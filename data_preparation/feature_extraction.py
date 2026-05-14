# When run from the command line this program takes in an audio file, a group label, and an output file and 
# extracts audio features from the audio file, then appends those features to the output file along with the group label.
# It expects the audio file to be in the data/audio/ folder and the output file to be in the data/output/ folder.
# To run from the command line: 
#    python -m analysis.feature_extraction --audiofile pedro_foley.wav --group 0 --outputfile output1.csv
# When run the first time, you may want to add the headers by changing the last line to header=True
# Then change it back to False before running again. Or save headers elsewhere and add them back in manually after

# It extracts these features:
# - Root mean square energy (RMS)
# - Zero crossing rate (ZCR)
# - Mel-frequency cepstral coefficients (MFCCs) - First 13 coefficients and their deltas
# - Spectral centroid (SC)
# - Spectral bandwidth (SBAN)

# The features are averaged across the whole audio file, and the min and max values are also included as features.

# Other programs can also call the extract_features function directly and pass in the audio file. Group label 
# is optional when called from other programs. If no output file is provided, the results will be returned to the calling function 
# but not saved to a csv file.

import argparse
import numpy as np
import librosa
import os
import sys
import pandas as pd

def extract_features(audiofile: str, group: int | None = None) -> pd.DataFrame:
  filepath = os.path.join("data/audio/", audiofile)
  # Load the audio file
  audio, sr = librosa.load(filepath, sr=16000)
  FRAME_LENGTH = 1024
  HOP_LENGTH = FRAME_LENGTH//4

  # Extract features
  rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
  zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
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
    'filename': audiofile,
    'group': group,
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
    'group',
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
  return results

# in another file
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--audiofile', type=str, default=None)
  parser.add_argument('--group', type=int, default=None)
  parser.add_argument('--outputfile', type=str, default=None)
  args = parser.parse_args()
  audiofile = args.audiofile
  group = args.group
  outputfile = args.outputfile

  filepath = os.path.join("data/audio/", audiofile)
  if audiofile is None:
    print("Please provide an audio file  using --audiofile")
    sys.exit(1)
  if not os.path.isfile(filepath):
    print(f"File {filepath} does not exist.")
    sys.exit(1)
  if group is None:
    print("Please provide a group using --group")
    sys.exit(1)
  if outputfile is None:
    print("No outputfile provided, results will be returned to the calling function but not saved to a csv file. To save results to a csv file, provide an outputfile name using --outputfile")
    
  results = extract_features(audiofile, group)
  if outputfile:
    results.to_csv(f"data/output/{outputfile}", mode="a", header=False, index=False)
