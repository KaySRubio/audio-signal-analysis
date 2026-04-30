import librosa
import librosa.display
from vad import EnergyVAD # https://pypi.org/project/energy-vad/
from typing import TypedDict, Literal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

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
def plot_spectrogram_zoom(Y: np.ndarray, sr: int, hop_length: int, y_axis: str ="mel", fmin: int =0, fmax: int =20000, xmin=0, xmax=12) -> tuple[Figure, Axes]:
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
   stft_filtered: np.ndarray
   stft_bat_filtered_abs: np.ndarray
   stft_filtered_dB: np.ndarray
# Apply a cutoff, and zero out all the values below (high-pass filter) or above (low-pass filter)
def ideal_frequency_filter(stft: np.ndarray, sr: int, frame_length: int, cutoff: int, filter: Literal['high', 'low']) -> FrequencyFilterReturn:
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
    stft_bat_filtered_abs = np.abs(stft_filtered) ** 2 # use for a power spectrogram

    # Transform amplitude OR power from linear to log scale decibels to improve visualization
    stft_filtered_dB = librosa.power_to_db(np.abs(stft_bat_filtered_abs), ref=np.max)

    result: FrequencyFilterReturn = {
       'stft_filtered': stft_filtered,
       'stft_bat_filtered_abs': stft_bat_filtered_abs,
       'stft_filtered_dB': stft_filtered_dB,
    }

    return result



