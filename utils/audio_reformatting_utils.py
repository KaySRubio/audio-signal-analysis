from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable
import librosa
import soundfile as sf
from functools import partial
from scipy.signal import resample
import re
import sox
import random

"""
Higher order function will find all files inside the path recursively, then apply the callback 
function to them and return a list of whatever the callback creates

Example uses is when you want to find all .txt files in a folder and create a dictionary
using properties extracted from the files. Then you can get a list of dictionaries and turn
it into a pandas df that holds metadata on the files

def turn_file_into_dict(file) -> dict:
  filename = file.name
  # Turn this text file into a df
  df = pd.read_csv(file, sep='\t')
  # do some calculations on columns of the df and get some properties
  return {
      "path": file,
      "filename": filename,
      ...
    }
file_data = find_all_files_and_apply_function(folder, ".txt", turn_file_into_dict)
df = pd.DataFrame(file_data)

Or you can get a list of .wav files in the parent folder
"""

def find_all_files_and_apply_function(
    path_to_directory: Path | str,
    file_extension: Literal[".xlsx", ".txt", ".csv", ".wav"],
    callback_func: Callable[[Path], Any],
) -> list:
    results = []
    path_to_directory = Path(
        path_to_directory
    )  # turn to a path in case user passed a string

    for file in path_to_directory.rglob("*" + file_extension):
        try:
            result = callback_func(file)
            results.append(result)

        except Exception as e:
            print("Error with file ", file.name, ":", e)

    return results

def read_wav_file_to_bytes(file_path: Path | str):
    with open(file_path, "rb") as wav_file:
        audio_bytes = wav_file.read()
    return audio_bytes

"""
Uses librosa to resamples a wav file and saves in the same directory with a modified filename indicating the new SR

Example use on one file: resample_wav(filepath, 48000)

Example use on a folder of files:
from functools import partial
find_all_files_and_apply_function(
    path_to_directory,
    ".wav",
    partial(resample_wav, sr=48000) # use partial so it doesn't get called immediately
)
"""

def resample_wav(filepath: Path | str, sr: int = 16000):
    print("resampling ", filepath, " to ", sr, "Hz")
    filepath = Path(filepath)
    audio, _ = librosa.load(filepath, sr=sr)  # loads and resamples
    filepath_without_ext = str(Path(filepath).with_suffix(""))
    sf.write(f"{filepath_without_ext}_{sr}Hz.wav", audio, sr)

"""
Uses soundfile to update SR, the number of channels, and bit depth

Example use on one file and save in same folder
reformat_wav(filepath, 16000, 1, "PCM_16")

Example use on one file and save in different folder (make sure already exists)
reformat_wav(
    '../../data/audio/test_files/background_noise/washing.wav',
    16000,
    1,
    "PCM_16",
    '../../data/audio/test_files/background_noise/reformatted',
)

Example use on multiple files:
find_all_files_and_apply_function(
    path_to_directory,
    ".wav",
    partial(reformat_wav, target_sr=16000, channels=1, subtype="PCM_16")
)
"""
def reformat_wav(
    filepath: Path | str,
    target_sr: int = 16000,
    channels: int = 1,
    subtype: Literal["PCM_16", "PCM_24"] = "PCM_16",
    output_directory: Path | str = "",
):
    print("Reformatting ", filepath, " to ", subtype, " and sr=", target_sr, "Hz")

    audio, sr = sf.read(filepath)
    if sr != target_sr:
        num_samples = round(len(audio) * target_sr / sr)
        audio = resample(audio, num_samples)

    # merge channels down to mono if requested
    if channels == 1 and audio.ndim > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1)

    output_path = ""
    if output_directory == "":
        filepath_without_ext = str(Path(filepath).with_suffix(""))
        output_path = f"{filepath_without_ext}_reformatted.wav"
    else:
        output_path = f"{output_directory}/{Path(filepath).stem!s}.wav"

    sf.write(
        output_path,
        audio,
        target_sr,
        subtype=subtype,
    )

"""
Use SoX and pySoX to get file information including:
 - channels
 - sample_rate
 - bitdepth (bits per sample)
 - bitrate (bytes per second)
 - num_samples
 - encoding (e.g., 'Signed Integer PCM')
 - silent (bool)
 - duration_s
 - duration_m
 - file_type (e.g., 'wav')
 - file_extension (e.g., 'wav')
* Note: must have SoX installed on your machine, e.g., brew install sox

Example use for 1 file
info = get_file_info(filepath)

Example use for multiple .wav files
file_info = find_all_files_and_apply_function(
    path_to_directory,
    ".wav",
    get_file_info
)
df = pd.DataFrame(file_info)
"""

def get_file_info(
    filepath: Path | str,
):
    try:
        # check if filepath is valid
        sox.file_info.validate_input_file(filepath)
        # use SoX to get data about the file
        info = sox.file_info.info(filepath)
        info["duration_s"] = info["duration"]
        info["duration_m"] = info["duration"] / 60
        info["file_type"] = sox.file_info.file_type(filepath)
        info["file_extension"] = sox.file_info.file_extension(filepath)
        info.pop("duration")
        # metadata about the file using the Path library
        filepath2 = Path(filepath)
        info["filename"] = filepath2.stem
        info["filepath"] = filepath
        # put the filename first in the object
        info = {"filename": info["filename"], **info}
        return info
    except (OSError, sox.core.SoxError) as e:
        print(filepath, " was not valid file:", e)
        return None

"""
Use SoX and pySoX to get audio file statistics including:
 - Samples_read
 - Length_(seconds)
 - Scaled_by
 - Maximum_amplitude
 - Minimum_amplitude
 - Midline_amplitude
 - Mean_norm
 - Mean_amplitude
 - RMS_amplitude
 - Maximum_delta
 - Minimum_delta
 - Mean_delta
 - RMS_delta
 - Rough_frequency
 - Volume_adjustment
* Note: must have SoX installed on your machine, e.g., brew install sox

Example use for 1 file
info = get_file_stats(filepath)

Example use for multiple .wav files
file_info = find_all_files_and_apply_function(
    path_to_directory,
    ".wav",
    get_file_stats
)
df = pd.DataFrame(file_stats)
"""

def get_file_stats(
    filepath: Path | str,
):
    try:
        # check if filepath is valid
        sox.file_info.validate_input_file(filepath)
        stats = sox.file_info.stat(filepath)
        # metadata about the file using the Path library
        filepath2 = Path(filepath)
        stats["filename"] = filepath2.stem
        stats["filepath"] = filepath
        # put the filename first in the object
        stats = {"filename": stats["filename"], **stats}
        # update the key names in this dictionary to have underscores rather than whitespace
        stats2 = {re.sub(r"\s+", "_", k): v for k, v in stats.items()}
        return stats2
    except (OSError, sox.core.SoxError) as e:
        print(filepath, " was not valid file:", e)
        return None

"""
Combines 2 files
Takes in 2 input files, and an output directory, as well as an argument for how 
to combine them and an optional list of volumes to apply
Make sure the files already have same format (sr, bit depth, channels)

Simple example use without changing volumes
input1 = '../../data/audio/test_files/160936251.wav'
input2 = '../../data/audio/test_files/thumps_reformatted.wav'
output_directory = '../../data/audio/test_files/merged_files' # make sure directory exists
combine_files(
    input1,
    input2,
    output_directory,
)

Methods to combine:
 - 'concatenate': combine input files by concatenating in the order given.
 - 'merge': combine input files by stacking each input file into a new channel
 - 'mix': combine input files by summing samples in corresponding channels
 - 'mix-power': combine input files with volume adjustments such that the output 
    volume is roughly equivlent to one of the input signals.
 - 'multiply': combine input files by multiplying samples
# for more information see https://github.com/marl/pysox/blob/master/sox/combine.py

Optional to list changes in volumes to be applied upon combining input files. 
input_volumes = [1, .5]

Volumes are applied to the input files in order. To keep volume the same, use 1.
To reduce by ~1 dB, use .9. To reduce by ~2dB, use .8
combine_files(
    input1,
    input2,
    output_directory,
    'mix',
    input_volumes
)
"""

def combine_files(
    input1: Path | str,
    input2: Path | str,
    output_directory: Path | str,
    combine_type: Literal[
        "concatenate", "merge", "mix", "merge-power", "multiply"
    ] = "mix",
    input_volumes: list[int] | None = None,
):
    try:
        # Create pysox Combiner class
        cbn = sox.Combiner()
        filename1 = Path(input1).stem
        filename2 = Path(input2).stem
        output = f"{output_directory}/{filename1}_{filename2}.wav"
        cbn.build([input1, input2], output, combine_type, input_volumes)
    except Exception as e:
        print("Error combining ", input1, " and ", input2, ": ", e)

'''
Combine multiple files. Takes in 1 input directories and 1 output directory
Iterates over each file in input_directory1, picks a random file from
input_directory2, and merges them, saving the new file in the output_directory
use input_volumes to adjust the volume of the first and/or second file in each merge
* Make sure all directories exist, and that all files are same format (sr, bit depth, channels)

Example use:
input_directory1 = '../../data/audio/test_files/clean_speech'
input_directory2 = '../../data/audio/test_files/background_noise'
output_directory = '../../data/audio/test_files/merged_files'
input_volumes = [1, .5]

combine_multiple_files(
  input_directory1,
  input_directory2,
  output_directory,
  'mix'
  input_volumes
)
'''
def combine_multiple_files(
    input_directory1: Path | str,
    input_directory2: Path | str,
    output_directory: Path | str,
    combine_type: Literal['concatenate', 'merge', 'mix', 'merge-power', 'multiply'] = 'mix',
    input_volumes: list[int] | None = None,
):
  input_directory1 = Path(input_directory1)
  input_directory2 = Path(input_directory2)
  # List all files in the second directory
  input_file2_list = list(input_directory2.rglob("*.wav"))

  for input_file1 in input_directory1.rglob("*.wav"):
    try:
      input_file2 = random.choice(input_file2_list)
      
      combine_files(
        input_file1,
        input_file2,
        output_directory,
        combine_type,
        input_volumes,
      )
        
    except Exception as e:
        print("Error during input_file ", input_file1, ": ", e)

