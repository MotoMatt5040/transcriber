import os
import re

import librosa
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence

from utils.logger_config import logger


def print_red(*args):
    message = ' '.join(map(str, args))
    print(f"\033[91m{message}\033[0m")


def print_light_blue(*args):
    message = ' '.join(map(str, args))
    print(f"\033[94m{message}\033[0m")


def print_light_green(*args):
    message = ' '.join(map(str, args))
    print(f"\033[92m{message}\033[0m")


def estimate_time(amount: int, iteration: int = 0, cuda_device_name: str = None) -> int:
    match cuda_device_name:
        case "NVIDIA GeForce RTX 4090":
            return round((amount - iteration) * 4)
        case "Tesla T4":
            return round((amount - iteration) * 8)
        case _:
            return round((amount - iteration) * 16)


def clean_text(text: str) -> list:
    """
    Clean text by removing commas, converting to lowercase, removing font tags, and splitting the text.
    """
    text = text.replace(',', '')
    text = text.lower()
    text = re.sub(r'{font.*?{/font}', '', text)
    text = text.strip()
    text = text.split()
    return text


def extract_segment(file_path, start_time, end_time, output_path):
    audio = AudioSegment.from_wav(file_path)
    start_ms = start_time * 1000
    end_ms = end_time * 1000
    segment = audio[start_ms:end_ms]
    segment.export(output_path, format="wav")


def segment_audio_by_silence(file_path, silence_thresh=-40, min_silence_len=2000):
    audio = AudioSegment.from_wav(file_path)
    segments = split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)

    segment_files = []
    for i, segment in enumerate(segments):
        segment_file_path = f"_segment_{i}.wav"
        segment.export(segment_file_path, format="wav")
        segment_files.append(segment_file_path)

    return segment_files


def convert_audio_format(input_file, output_file, format='wav'):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format=format)


def normalize_audio(audio):
    """Normalize an AudioSegment in memory."""
    return audio.normalize()


def resample_audio(audio, target_sample_rate=16000):
    """Resample an AudioSegment in memory."""
    if audio.frame_rate != target_sample_rate:
        return audio.set_frame_rate(target_sample_rate)
    return audio


def trim_silence_audio(audio, silence_thresh=-40, min_silence_len=2000):
    """Trim silence from an AudioSegment in memory. Returns original audio if no segments found."""
    trimmed_segments = split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)
    if not trimmed_segments:
        return audio
    combined = AudioSegment.silent(duration=0)
    for segment in trimmed_segments:
        combined += segment + AudioSegment.silent(duration=500)
    return combined


def noise_reduce_audio(audio):
    """Apply noise reduction to an AudioSegment in memory. Returns a new AudioSegment."""
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    sr = audio.frame_rate
    reduced_noise = nr.reduce_noise(y=samples, sr=sr)
    reduced_audio = AudioSegment(
        np.int16(reduced_noise * 32767).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    return reduced_audio


def preprocess_audio(file_path):
    """
    Preprocess audio in memory: convert, normalize, resample, noise reduce, trim silence.
    Writes only a single output file instead of multiple intermediate files.
    Returns the path to the processed file.
    """
    audio = AudioSegment.from_file(file_path)
    audio = normalize_audio(audio)
    audio = resample_audio(audio, target_sample_rate=16000)
    audio = noise_reduce_audio(audio)
    audio = trim_silence_audio(audio)

    output_path = "_processed.wav"
    audio.export(output_path, format="wav")
    return output_path


def get_audio_length(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000
    return duration_seconds


def get_sentence_case(source: str) -> str:
    output = []
    is_first_word = True

    for c in source:
        if is_first_word and not c.isspace():
            c = c.upper()
            is_first_word = False
        elif not is_first_word and c in ".!?":
            is_first_word = True

        output.append(c)

    return ''.join(output)


def cleanup_temp_files(*paths):
    """Remove temporary files, ignoring errors if they don't exist."""
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.warning(f"Failed to clean up temp file {path}: {e}")
