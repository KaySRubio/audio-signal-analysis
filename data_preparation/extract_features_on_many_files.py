# This program is designed to be run from the command line and takes in a CSV file with two columns: 'filename' and 'group'.
# For each row in the CSV file, it calls the feature_extraction program to extract audio features from the specified audio file 
# and append those features to an output CSV file along with the group label.

# To run from the command line: 
# Make sure your metadata.csv file is saved in data/ folder and contains 2 columns with headers 'filename' and 'group'
#    python -m analysis.extract_features_on_many_files --csvfile metadata_clean_short.csv

# Header Issues:
# It's helpful if you first run feature_extraction.py on a single file to generate the headers in output.csv
# the save the headers in a separate file, and delete them in output.csv before running this script, then put them back
# in manually afterwards. Otherwise weird stuff happens with the first row getting written to the right of the headers

import subprocess
import pandas as pd
import argparse
import sys
import os

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--csvfile', type=str, default=None)
  args = parser.parse_args()
  csvfile = args.csvfile
  filepath = os.path.join("data/", csvfile)
  if csvfile is None:
    print("Please provide a CSV file using --csvfile")
    sys.exit(1)
  if not os.path.isfile(filepath):
    print(f"File {filepath} does not exist.")
    sys.exit(1)
  
  df = pd.read_csv(filepath)
  errors = []

  for _, row in df.iterrows():
      print(f"Processing {row['filename']}...")
      result = subprocess.run(
        ['python', '-m', 'analysis.feature_extraction',
        '--audiofile', row['filename'],
        '--group', str(row['group']),
        '--outputfile', 'output.csv'],
        capture_output=True,
        text=True
      )
      if result.returncode != 0:
          print(f"Error on {row['filename']}: {result.stderr}")
          errors.append(row['filename'])
      else:
          print(f"Done: {row['filename']}")
  if errors:
     print("Errors occurred for the following files: " + ", ".join(errors))
