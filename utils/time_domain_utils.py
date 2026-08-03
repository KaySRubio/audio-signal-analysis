import librosa
import librosa.display
from vad import EnergyVAD # https://pypi.org/project/energy-vad/
from typing import TypedDict, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import math
import torch
import soundfile as sf
from scipy.io import wavfile

class Timestamp(TypedDict):
    start: float
    end: float

# Plot waveform with optional regions highlighted using timestamps
def plot_waveform(
    audio: np.ndarray,
    title: str,
    sr: int | float = 16000,
    sections: list[Timestamp] | None = None,
    sections2: list[Timestamp] | None = None
) -> tuple[Figure, Axes]:

    sections = sections or []
    sections2 = sections2 or []

    fig, ax = plt.subplots(figsize=(10, 4))

    librosa.display.waveshow(audio, sr=sr, alpha=0.5, color="blue", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude")

    for i, section in enumerate(sections):
        start, end = float(section["start"]), float(section["end"])
        ax.axvspan(start, end, color="purple", alpha=0.3)

        # Put label in the middle of the region
        x = (start + end) / 2
        y = ax.get_ylim()[1] * 0.9

        ax.text(
            x,
            y,
            f"{i+1}",
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )


    for section in sections2:
        start, end = float(section["start"]), float(section["end"])
        ax.axvspan(start, end, color="yellow", alpha=0.3)

    return fig, ax


# Function to plot 1 waveform with optional line
# color options include r, g, 
def plot_waveform_with_line(
    audio: np.ndarray,
    title: str,
    hop_length: int,
    sr: int | float = 16000,
    lineValues: list[int] | np.ndarray = [],
    color="r"
):
    frames = range(len(lineValues))
    t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(15, 3))
    librosa.display.waveshow(audio, sr=sr, alpha=0.5)
    plt.plot(t, lineValues, color=color)
    plt.title(title)
    plt.xlabel("Time (sec)") 
    plt.ylabel("Amplitude")
    plt.show()

# Function to plot multiple waveforms in 1 figure
# audios and titles should be arrays of equal length
# also assumes sr is already defined as variable sr
def plot_waveforms(audios: list[np.ndarray], titles: list[str], sr: int | float = 16000):
    if(len(audios) != len(titles)):
        raise ValueError(f"Error: audios and titles should be arrays of the same length")
    num_of_waves = len(audios)
    plt.figure(figsize=(15, 8))
    for i, audio in enumerate(audios):
        plt.subplot(num_of_waves, 1, i+1) # create subplots of 3 files
        librosa.display.waveshow(audio, sr=sr, alpha=0.5)
        plt.title(titles[i])
    plt.subplots_adjust(hspace=0.5)  # Increase vertical spacing for titles
    plt.show()

# Plot waveform with optional regions highlighted using timestamps and include numeric labels
def plot_waveform_with_labels(
    audio: np.ndarray,
    title: str,
    sr: int | float = 16000,
    sections: list[Timestamp] | None = None,
    sections2: list[Timestamp] | None = None,
    label_sections: bool = False,
) -> tuple[Figure, Axes]:

    sections = sections or []
    sections2 = sections2 or []

    fig, ax = plt.subplots(figsize=(10, 4))

    librosa.display.waveshow(audio, sr=sr, alpha=0.5, color="blue", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Amplitude")

    for i, section in enumerate(sections):
        start, end = float(section["start"]), float(section["end"])
        ax.axvspan(start, end, color="purple", alpha=0.3)

        if label_sections:
          # Put label in the middle of the region
          x = (start + end) / 2
          # Stagger labels to make them easier to read
          if i % 3 == 0:
            y = ax.get_ylim()[1] * 0.9
          elif i % 3 == 1:
            y = ax.get_ylim()[1] * 0.8
          else:
            y = ax.get_ylim()[1] * 0.7

          ax.text(
              x,
              y,
              f"{i}",
              ha="center",
              va="center",
              fontsize=10,
              color="black"
          )


    for i, section in enumerate(sections2):
        start, end = float(section["start"]), float(section["end"])
        ax.axvspan(start, end, color="yellow", alpha=0.3)

        if label_sections:
          # Put label in the middle of the region
          x = (start + end) / 2
          # Stagger labels to make them easier to read
          if i % 3 == 0:
            y = ax.get_ylim()[1] * 0.9
          elif i % 3 == 1:
            y = ax.get_ylim()[1] * 0.8
          else:
            y = ax.get_ylim()[1] * 0.7

          ax.text(
              x,
              y,
              f"{i}",
              ha="center",
              va="center",
              fontsize=10,
              color="black"
          )

    return fig, ax

# normalize audio signal so volume is consistent across files
def rms_normalize(y: np.ndarray, target_dBFS:float = -30.0) -> np.ndarray:
    rms = np.sqrt(np.mean(y**2))
    current_dBFS = 20 * np.log10(rms + 1e-9)
    gain = 10 ** ((target_dBFS - current_dBFS) / 20)
    return y * gain

# Function that takes in
#  - vad_output as a List of 0's and 1's, where 0 = frame with silence and 1 = voice activity
#  - frame_shift in ms
#  - min_silence_duration, which is the minimum time of silence in ms that separates a group of sound
# Returns periods of time with voice activity separated by silences longer than the min_silence_duration
# Output format: [{'start': number, 'end': number}]
def get_voice_activity_timestamps(*, vad_output: np.ndarray, frame_shift: int, min_silence_duration: int = 1000) -> list[Timestamp]:
    min_silence_frames = int(min_silence_duration / frame_shift)

    groups = []
    start_idx = None
    silence_counter = 0

    for i, frame in enumerate(vad_output):
        if frame == 1:
            if start_idx is None:
                start_idx = i
            silence_counter = 0
        else:
            if start_idx is not None:
                silence_counter += 1
                if silence_counter >= min_silence_frames:
                    # Silence is long enough, so close the current voice group
                    end_idx = i - silence_counter
                    start_time = start_idx * frame_shift / 1000
                    end_time = (end_idx + 1) * frame_shift / 1000
                    groups.append({
                        'start': round(start_time, 4),
                        'end': round(end_time, 4)
                    })
                    start_idx = None
                    silence_counter = 0

    # Handle case where audio ends with voice activity
    if start_idx is not None:
        end_time = (len(vad_output)+1) * frame_shift / 1000
        groups.append({
            'start': round(start_idx * frame_shift / 1000, 4),
            'end': round(end_time, 4)
        })

    return groups

# Function that takes in
#  - vad_output as a List of 0's and 1's, where 0 = frame with silence and 1 = voice activity
#  - frame_shift in ms
#  - min_silence_duration, which is the minimum time of silence in ms that will be included
# Returns timestamps for silences longer than the min_silence_duration
# Output format: [{'start': number, 'end': number}]
def get_timestamps_silences(*, vad_output: np.ndarray, frame_shift: int, min_silence_duration: int=1000) -> list[Timestamp]:
    min_silence_frames = int(min_silence_duration / frame_shift)
    silence_timestamps = []
    start_idx = None

    for i, frame in enumerate(vad_output):
        if frame==0:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                end_idx = i
                duration = end_idx - start_idx

                if duration >= min_silence_frames:
                    start_time = start_idx * frame_shift / 1000
                    end_time = end_idx * frame_shift / 1000
                    silence_timestamps.append({
                        'start': round(start_time, 2),
                        'end': round(end_time, 2)
                    })

                start_idx = None

    # Handle case where the last segment goes to the end
    if start_idx is not None:
        end_idx = len(vad_output)
        duration = end_idx - start_idx
        if duration >= min_silence_frames:
            start_time = start_idx * frame_shift / 1000
            end_time = end_idx * frame_shift / 1000
            silence_timestamps.append({
                'start': round(start_time, 2),
                'end': round(end_time, 2)
            })

    return silence_timestamps

# Convert timestamps into durations
# Returns timestamps in format {'start': time in seconds, 'end': time in seconds}
def convert_timestamps_to_durations(timestamps: list[Timestamp]) -> list[int]:
    durations = []
    for i, timestamp in enumerate(timestamps):
        durations.append(round(float(timestamp['end'])-float(timestamp['start']), 4))
    return durations

# Function to spit an audio into an array of individual audios by timestamp.
# Assumes timestamps are in format {'start': time in seconds, 'end': time in seconds}
def split_audio_by_timestamps(audio: np.ndarray, timestamps: list[Timestamp], sr: int | float = 16000) -> list[np.ndarray]:
    audio_array: list[np.ndarray] = []
    for i, ts in enumerate(timestamps):
        start_sample = int(float(ts['start']) * sr) # convert start time into sample index
        end_sample = int(float(ts['end']) * sr)
        segment = audio[start_sample:end_sample] # Extract the segment using start and ending sample index
        audio_array.append(segment) # append segment into the array
    return audio_array

# Function to calculate amplitude envelope for each frame
def amplitude_envelope(audio: np.ndarray, frame_size: int, hop_length: int) -> np.ndarray:
    return np.array([max(audio[i:i+frame_size]) for i in range(0, len(audio), hop_length)])

DITHER_VALUE = 0.05
MIN_AMPLITUDE_THRESHOLD = 1e-10
MIN_RMS_THRESHOLD = 1e-12
FALLBACK_RMS = 1e-6
SNR_DB_RANGE = torch.arange(-20, 101, dtype=torch.float32)
ALPHA_04_GAMMA_CURVE = torch.tensor(
    [
        0.409747739,
        0.409869263,
        0.409985656,
        0.409690892,
        0.409861864,
        0.409990055,
        0.410271377,
        0.410526266,
        0.411010238,
        0.411432644,
        0.412317178,
        0.413372716,
        0.415264259,
        0.417819198,
        0.420772515,
        0.424527992,
        0.429188858,
        0.435103734,
        0.442341951,
        0.451614855,
        0.462211529,
        0.474916474,
        0.488838093,
        0.505092356,
        0.523537093,
        0.543720882,
        0.565324274,
        0.588475317,
        0.613462118,
        0.639544959,
        0.667508177,
        0.695837243,
        0.724547622,
        0.754147993,
        0.783231484,
        0.81240985,
        0.842197752,
        0.871664058,
        0.900305039,
        0.928804177,
        0.95655449,
        0.983534905,
        1.010471548,
        1.0362095,
        1.061364248,
        1.085793118,
        1.109481904,
        1.132779949,
        1.154728256,
        1.176273084,
        1.197035028,
        1.216716938,
        1.235358982,
        1.253643127,
        1.271038908,
        1.287180295,
        1.303028647,
        1.318395272,
        1.332948173,
        1.347009353,
        1.360572696,
        1.373455135,
        1.385771224,
        1.397335037,
        1.408563968,
        1.41959619,
        1.42983624,
        1.439584667,
        1.449021764,
        1.458048307,
        1.466695685,
        1.474869384,
        1.48269965,
        1.490343394,
        1.49748214,
        1.504351061,
        1.510764265,
        1.516989146,
        1.522909703,
        1.528578001,
        1.533898351,
        1.539121095,
        1.543906502,
        1.54858517,
        1.553107762,
        1.557443906,
        1.561649273,
        1.565663481,
        1.569386712,
        1.573077668,
        1.576547638,
        1.57980083,
        1.583041292,
        1.586024961,
        1.588806813,
        1.591624771,
        1.594196895,
        1.596931549,
        1.599446005,
        1.601850111,
        1.604086681,
        1.60627134,
        1.608261987,
        1.610045475,
        1.611924722,
        1.61369656,
        1.615340743,
        1.616889049,
        1.618389159,
        1.619853744,
        1.621358779,
        1.622681189,
        1.623904229,
        1.625131432,
        1.626324628,
        1.6274027,
        1.628427675,
        1.629455321,
        1.6303307,
        1.631280263,
        1.632041021,
    ],
    dtype=torch.float32,
)

# Function to calculate SNR
def get_snr(filepath):
    """
    Calculate the WADA SNR value (in dB) for a given WAV audio tensor.
    Note: It's up to the caller to ensure the audio is in the correct format (WAV 16kHz, 16-bit).

    Args:
        audio (torch.Tensor): Input WAV tensor.
        dither_value (float): Dither amplitude scaling factor (default: 0.05).

    Returns:
        -99 for an error
        float: Estimated SNR value in decibels (dB).
        -	above 40 is very good, clean audio, almost no background noise
        -	30-40 clean enough, broadcast quality, minor noise
        -	20-30 moderate noise, still OK for most speech recognition
        -	10-20 noisy, speech should still be intelligible tho
        -	0 - 10 very noisy, hard to understand
    """
    
    try:
        file1_raw, sr_original = sf.read(filepath, dtype='int16')
        if sr_original != 16000:
            print(f"Error! SR must be 16000 but file found at {filepath} has sr of {sr_original}")
            return -99
        else:
            # Convert numpy array to torch tensor
            audio = torch.from_numpy(file1_raw).float()

            if audio.ndim == 0 or audio.numel() == 0:
                raise ValueError("Input audio tensor cannot be scalar or empty")

            # Convert to mono if multi channel
            if audio.ndim > 1 and audio.shape[0] > 1:
                audio = audio.mean(dim=0)

            dithered_audio = _add_dither(audio, DITHER_VALUE)
            centered_audio = _remove_dc_offset(dithered_audio)
            rectified_audio = _ensure_min_amplitude(centered_audio.abs())

            gamma = _compute_gamma_statistic(rectified_audio)
            snr_db = _map_gamma_to_snr(gamma)
            return snr_db
    except Exception as e:
        print(f"Error calculating WADA SNR: {e}")
        return -99

def _calculate_rms(audio: torch.Tensor) -> torch.Tensor:
    """Compute RMS (root mean square) of the audio signal."""
    rms = torch.sqrt(torch.mean(audio**2))
    if rms.item() < MIN_RMS_THRESHOLD:
        return torch.tensor(FALLBACK_RMS)
    return rms


def _add_dither(audio: torch.Tensor, dither_value: float) -> torch.Tensor:
    """Add RMS-scaled Gaussian noise to stabilize WADA estimation."""
    rms = _calculate_rms(audio)
    noise = torch.randn_like(audio) * dither_value * rms
    return audio + noise


def _remove_dc_offset(audio: torch.Tensor) -> torch.Tensor:
    """Remove DC offset (mean bias) from the waveform."""
    return audio - audio.mean()


def _ensure_min_amplitude(audio: torch.Tensor) -> torch.Tensor:
    """Ensure amplitudes are above a safe numeric threshold."""
    audio[audio < MIN_AMPLITUDE_THRESHOLD] = MIN_AMPLITUDE_THRESHOLD
    return audio


def _compute_gamma_statistic(audio: torch.Tensor) -> float:
    """Compute the WADA gamma statistic from absolute amplitudes."""
    d_val1 = audio.mean().item()
    d_val2 = torch.log(audio).mean().item()
    return math.log(d_val1) - d_val2


def _map_gamma_to_snr(v3: float) -> float:
    """Map gamma statistic to interpolated SNR (in dB) using lookup curve."""
    if v3 <= ALPHA_04_GAMMA_CURVE[0]:
        return float(SNR_DB_RANGE[0])
    if v3 >= ALPHA_04_GAMMA_CURVE[-1]:
        return float(SNR_DB_RANGE[-1])

    idx = torch.searchsorted(ALPHA_04_GAMMA_CURVE, torch.tensor(v3)) - 1
    idx = max(0, min(idx, len(SNR_DB_RANGE) - 2))

    gamma_lower = ALPHA_04_GAMMA_CURVE[idx]
    gamma_upper = ALPHA_04_GAMMA_CURVE[idx + 1]
    db_lower = SNR_DB_RANGE[idx]
    db_upper = SNR_DB_RANGE[idx + 1]

    interp = (v3 - gamma_lower) / (gamma_upper - gamma_lower)
    return float(db_lower + interp * (db_upper - db_lower))

def show_audio_info(input_filepath):
    rate, data = wavfile.read(input_filepath)
    all_numbers = data.flatten()
    channels = 1 if data.ndim == 1 else data.shape[1]

    print(input_filepath)
    print(f'sampling rate: {rate}')
    print(f'data: {data}')
    print()
    print(f'data.shape: {data.shape}')
    print(f'channels:       {channels}')
    print(f'data.dtype: {data.dtype}')
    print(f'duration:       {data.shape[0] / rate:.2f} seconds')
    print(f'max:            {all_numbers.max():.4f}')
    print(f'min:            {all_numbers.min():.4f}')
    print(f'mean (DC offset):      {all_numbers.mean():.6f}')

def show_array_info(data):
    all_numbers = data.flatten()
    channels = 1 if data.ndim == 1 else data.shape[1]

    print(f'data: {data}')
    print()
    print(f'data.shape: {data.shape}')
    print(f'channels:       {channels}')
    print(f'data.dtype: {data.dtype}')
    print(f'max:            {all_numbers.max():.4f}')
    print(f'min:            {all_numbers.min():.4f}')
    print(f'mean (DC offset):      {all_numbers.mean():.6f}')
