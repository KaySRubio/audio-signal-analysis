import librosa
import librosa.display
from vad import EnergyVAD # https://pypi.org/project/energy-vad/
from typing import TypedDict, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

class Timestamp(TypedDict):
    start: float
    end: float


def plot_waveform(
    audio: np.ndarray,
    title: str,
    sr: int = 16000,
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

    for section in sections:
        start, end = float(section["start"]), float(section["end"])
        ax.axvspan(start, end, color="purple", alpha=0.3)

    for section in sections2:
        start, end = float(section["start"]), float(section["end"])
        ax.axvspan(start, end, color="yellow", alpha=0.3)

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
def split_audio_by_timestamps(audio: np.ndarray, timestamps: list[Timestamp], sr: int = 16000) -> list[np.ndarray]:
    audio_array: list[np.ndarray] = []
    for i, ts in enumerate(timestamps):
        start_sample = int(float(ts['start']) * sr) # convert start time into sample index
        end_sample = int(float(ts['end']) * sr)
        segment = audio[start_sample:end_sample] # Extract the segment using start and ending sample index
        audio_array.append(segment) # append segment into the array
    return audio_array