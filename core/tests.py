import torch
import os
import whisper
import time
import re

import librosa
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pyannote.audio import Pipeline

from utils.logger_config import logger

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ['hugging_face_token'])

# TODO Test with Q17OE 12948 12751
# //ippronto1.colo1.promarkresearch.com/vox/12943PCM/Q4OE_0000340508.wav
# //ippronto2.colo1.promarkresearch.com/vox/12943PCM/Q4OE_0000340508.wav
# //ippronto1.colo1.promarkresearch.com/vox/12948PCM/Q17OE_000012751.wav

cuda_is_available = torch.cuda.is_available()
cuda_device_name = None
if cuda_is_available:
    torch.cuda.init()
    logger.info(f"CUDA is available {cuda_is_available}")
    logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device {torch.cuda.current_device()}")
    logger.info(f"Current CUDA device address {torch.cuda.device(0)}")
    cuda_device_name = torch.cuda.get_device_name(0)
    logger.info(f"Current CUDA device name: {cuda_device_name}")
    logger.info(torch.version.cuda)

    torch.set_default_device('cuda')
    pipeline.to(torch.device('cuda'))

else:
    logger.warning("CUDA is not available, using CPU instead")
    torch.set_default_device('cpu')


model = whisper.load_model("medium", device="cuda")
model.to(torch.device('cuda'))


def print_red(*args):
    message = ' '.join(map(str, args))
    print(f"\033[91m{message}\033[0m")


def print_light_blue(*args):
    message = ' '.join(map(str, args))
    print(f"\033[94m{message}\033[0m")


def print_light_green(*args):
    message = ' '.join(map(str, args))
    print(f"\033[92m{message}\033[0m")


def estimate_time(amount: int, iteration: int = 0) -> int:
    match cuda_device_name:
        case "NVIDIA GeForce RTX 4090":
            return round((amount - iteration) * 4)
        case "Tesla T4":
            return round((amount - iteration) * 8)
        case _:
            return round((amount - iteration) * 16)


def clean_text(text: str) -> list:
    text = text.replace(',', '')
    text = text.lower()
    text = re.sub(r'{font.*?{/font}', '', text)
    text = text.strip()
    text = text.split()
    return text


def extract_segment(file_path, start_time, end_time, output_path):
    audio = AudioSegment.from_wav(file_path)
    start_ms = start_time * 1000  # Convert to milliseconds
    end_ms = end_time * 1000  # Convert to milliseconds
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


def normalize_audio(input_file, output_file):
    audio = AudioSegment.from_wav(input_file)
    normalized_audio = audio.normalize()
    normalized_audio.export(output_file, format='wav')


def resample_audio(input_file, output_file, target_sample_rate=16000):
    audio = AudioSegment.from_wav(input_file)
    if audio.frame_rate != target_sample_rate:
        resampled_audio = audio.set_frame_rate(target_sample_rate)
        resampled_audio.export(output_file, format='wav')
    else:
        audio.export(output_file, format='wav')


def trim_silence(input_file, output_file, silence_thresh=-40, min_silence_len=2000):
    audio = AudioSegment.from_wav(input_file)
    trimmed_audio = split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)
    combined_audio = AudioSegment.silent(duration=0)
    for segment in trimmed_audio:
        combined_audio += segment + AudioSegment.silent(duration=500)
    combined_audio.export(output_file, format='wav')


def noise_reduce_audio(input_file, output_file):
    y, sr = librosa.load(input_file, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    reduced_audio = AudioSegment(
        np.int16(reduced_noise * 32767).tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )
    reduced_audio.export(output_file, format="wav")


def preprocess_audio(file_path):
    temp_file = f"_processed.wav"
    convert_audio_format(file_path, temp_file)
    normalized_file = f"{temp_file}_normalized.wav"
    normalize_audio(temp_file, normalized_file)
    resampled_file = f"{normalized_file}_resampled.wav"
    resample_audio(normalized_file, resampled_file)
    noise_reduced_file = f"{resampled_file}_noise_reduced.wav"
    noise_reduce_audio(resampled_file, noise_reduced_file)
    trimmed_file = f"{resampled_file}_trimmed.wav"
    trim_silence(resampled_file, trimmed_file)

    return trimmed_file


def get_audio_length(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000
    return duration_seconds


def get_sentence_case(source: str):
    output = ""
    is_first_word = True
    for c in source:
        if is_first_word and not c.isspace():
            c = c.upper()
            is_first_word = False
        elif not is_first_word and c in ".!?":
            is_first_word = True
        output = output + c
    return output


class Transcribe:
    def __init__(self, model: str):
        print(model)
        self._model = whisper.load_model(model, device="cuda")
        self._model.to(torch.device('cuda'))
        self.transcription_dict = {}
        self.transcription_errors = []

    def transcribe(self):
        assert self.model is not None, "Model is not loaded"

        file_path = r"tests\audio\QINITIALOE_0000048840.wav"
        processed_file = preprocess_audio(file_path)
        segment_files = segment_audio_by_silence(processed_file)
        current_sentence = ''

        interviewer = None
        # TODO fix repeats using 12926 Q20_1OE 0000044033
        for segment_file in segment_files:
            diarization = pipeline(segment_file, min_speakers=2, max_speakers=2)
            segment_transcriptions = []
            prev_speaker = None
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                audio_segment_path = f"_segment_{turn.start}_{turn.end}.wav"
                extract_segment(segment_file, turn.start, turn.end, audio_segment_path)
                segment_transcription = model.transcribe(audio_segment_path, language='en')
                if not prev_speaker:
                    prev_speaker = speaker
                    segment_transcriptions.append(f"{speaker}: {segment_transcription['text']}")
                    current_sentence += f"{speaker}: {segment_transcription['text']}"
                else:
                    if prev_speaker == speaker:
                        segment_transcriptions.append(segment_transcription['text'])
                        current_sentence += segment_transcription['text']
                    else:
                        prev_speaker = speaker
                        segment_transcriptions.append(f"{speaker}: {segment_transcription['text']}")
                        current_sentence = f"{speaker}: {segment_transcription['text']}"

                os.remove(audio_segment_path)

        speaker_transcription = model.transcribe(file_path, language='en')['text']
        transcription = get_sentence_case(speaker_transcription.strip())
        print(transcription)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = whisper.load_model(m)



