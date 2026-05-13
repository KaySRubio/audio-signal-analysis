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


# Function to extract Short-Time Fourier Transform
def short_time_fourier_transform(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
    type: Literal["power", "amplitude"] = "power"
) -> np.ndarray:
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length) # outputs matrix of complex numbers for stft
    # stft.shape # optional - check shape
    # type(stft[0][0]) # optional - check output type, should be complex number containing phase & amplitude
    
    # Remove complex numbers to retain just amplitude/power, and transform to decibels to improve visualization
    if type=="power":
        stft_abs = np.abs(stft) ** 2 # use for a power spectrogram
        stft_dB = librosa.power_to_db(np.abs(stft_abs), ref=np.max)
    elif type=="amplitude":
        stft_abs = np.abs(stft) # use for an amplitude spectrogram
        stft_dB = librosa.amplitude_to_db(np.abs(stft_abs), ref=np.max)
    else:
        raise ValueError(f"Error: type should be 'power' or 'amplitude'")
    # type(stft_abs[0][0]) # optional - check to ensure no longer complex
    return stft_dB

# Create method to easily plot spectrograms
def plot_spectrogram(Y: np.ndarray, sr: int | float, hop_length: int, y_axis: str ="mel", fmin: int = 0, fmax: int = 20000) -> tuple[Figure, Axes]:
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
def plot_spectrogram_zoom(Y: np.ndarray, sr: int | float, hop_length: int, y_axis: str ="mel", fmin: int =0, fmax: int =20000, xmin: float=0, xmax: float=12) -> tuple[Figure, Axes]:
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
      sr: int | float,
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

# Function to plot spectral centroid and bandwidth
def plot_spectral_centroid(
    sc: np.ndarray,
    sban: np.ndarray,
    hop_length: int,
    title: str = ''
):
    frames = range(len(sc))
    time = librosa.frames_to_time(frames, hop_length=hop_length) # TODO --> may need to add sr=sr
    # Calculate upper and lower bounds
    upper = np.array(sc) + np.array(sban)
    lower = np.array(sc) - np.array(sban)
    plt.figure(figsize=(25,10))
    plt.plot(time, sc, color='b', label='Spectral Centroid')
    plt.fill_between(time, upper, lower, alpha=0.2, label='Spectral Bandwidth')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spectral centroid values')
    plt.title(title)
    plt.legend()
    plt.show()

# Visualizing MFCCs
def plot_mfccs(mfccs: np.ndarray, sr: int | float, title: str = ''):
  plt.figure(figsize=(25, 5))
  librosa.display.specshow(mfccs, x_axis='time', sr=sr)
  plt.colorbar(format="%+2.f")
  plt.title(title)
  plt.tight_layout()
  plt.show()