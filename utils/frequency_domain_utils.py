import librosa
import librosa.display
from vad import EnergyVAD # https://pypi.org/project/energy-vad/
from typing import TypedDict, Literal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

class FrequencyData(TypedDict):
    min_freq: float
    max_freq: float
    max_energy_freq: float

# Create method to easily plot spectrograms
def plot_spectrogram(Y: np.ndarray, sr: int, hop_length: int, y_axis: str ="mel", fmin: int = 0, fmax: int = 20000) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(25, 4))
    
    #plt.figure(figsize=(25, 10)) # instantiate a figure and give a size
    # use librosa.display.specshow to visualize any type of spectrogram
    librosa.display.specshow(Y, 
          sr=sr, 
          hop_length=hop_length, 
          x_axis="time", 
          y_axis=y_axis
		)
    ax.set_ylim(fmin, fmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')    
    # fig.set_colorbar(format="%+2.f") # add a color bar legend
    ax.set_title('Spectrogram')
    return fig, ax

# Create method to easily plot spectrograms AND zoom in on the x-axis
def plot_spectrogram_zoom(Y: np.ndarray, sr: int, hop_length: int, y_axis: str ="mel", fmin: int =0, fmax: int =20000, xmin: float=0, xmax: float=12) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(25, 10)) # instantiate a figure and give a size
    # use librosa.display.specshow to visualize any type of spectrogram
    librosa.display.specshow(Y, 
      sr=sr, 
      hop_length=hop_length, 
      x_axis="time", 
      y_axis=y_axis
		)
    ax.set_ylim(fmin, fmax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')    
    # plt.colorbar(format="%+2.f") # add a color bar legend
    ax.set_title('Spectrogram')
    return fig, ax

class FrequencyFilterReturn(TypedDict):
   frequencies: np.ndarray
   stft_filtered: np.ndarray
   stft_filtered_abs: np.ndarray
   stft_filtered_dB: np.ndarray
# Apply a cutoff, and zero out all the values below (high-pass filter) or above (low-pass filter)
def ideal_frequency_filter(
      stft: np.ndarray,
      sr: int,
      frame_length: int,
      cutoff: int,
      filter: Literal['high', 'low']
) -> FrequencyFilterReturn:
    # Get the frequency values corresponding to the rows of the STFT
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length)

    # Find the index of the frequency closest to the cutoff
    cutoff_idx = np.argmax(frequencies >= cutoff)

    # Create a copy of the STFT matrix
    stft_filtered = stft.copy()

    # Zero out frequencies below cutoff
    if filter == 'high':
      stft_filtered[:cutoff_idx, :] = 0
    else:
      stft_filtered[cutoff_idx:, :] = 0

    # Remove complex numbers, retain just amplitude, and choose between amplitude or power spectrograms
    stft_filtered_abs = np.abs(stft_filtered) ** 2 # use for a power spectrogram

    # Transform amplitude OR power from linear to log scale decibels to improve visualization
    stft_filtered_dB = librosa.power_to_db(np.abs(stft_filtered_abs), ref=np.max)

    result: FrequencyFilterReturn = {
       'frequencies': frequencies,
       'stft_filtered': stft_filtered,
       'stft_filtered_abs': stft_filtered_abs,
       'stft_filtered_dB': stft_filtered_dB,
    }

    return result

# Function that takes in:
    #   stft_abs which comes from stft = librosa.stft() && stft_abs = np.abs(stft_bat) 
    #   and where filters for background noise have already been applied
    #   times, which comes from librosa.frames_to_time()
    #   the start and end times (in seconds) of the call
    #   an energy cutoff, for instance, what's the minimum energy that something has to have
    #   to be considered part of the sound
# Function returns the min frequency, the max frequency, and the frequency with most energy (Hz)
def extract_min_max_and_most_energy_frequencies(
    stft_abs: np.ndarray,
    times: np.ndarray,
    start_time: float,
    end_time: float,
    energy_cutoff: float,
    frequencies: np.ndarray
) -> FrequencyData:
    # slice time window for the first call
    time_mask = (times >= start_time) & (times <= end_time)
    stft_call = stft_abs[:, time_mask]
    
    # energy per frequency bin
    energy_per_freq = stft_call.mean(axis=1)

    # normalize
    energy_norm = energy_per_freq / energy_per_freq.max()

    # cutoff mask
    mask = energy_norm >= energy_cutoff

    # get results
    min_freq = frequencies[mask].min()
    max_freq = frequencies[mask].max()
    max_energy_freq = frequencies[np.argmax(energy_per_freq)]

    # return results
    return {'min_freq': min_freq, 'max_freq': max_freq, 'max_energy_freq': max_energy_freq}



