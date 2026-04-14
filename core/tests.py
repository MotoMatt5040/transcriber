import torch
import os
import whisper

from pyannote.audio import Pipeline

from utils.logger_config import logger
from core.audio_utils import (
    extract_segment, segment_audio_by_silence,
    preprocess_audio, get_sentence_case,
)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ['hugging_face_token'])

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


class Transcribe:
    def __init__(self, model: str):
        print(model)
        self._model = whisper.load_model(model, device="cuda")
        self._model.to(torch.device('cuda'))

    def transcribe(self):
        assert self.model is not None, "Model is not loaded"

        file_path = r"tests\audio\QINITIALOE_0000048840.wav"
        processed_file = preprocess_audio(file_path)
        segment_files = segment_audio_by_silence(processed_file)
        current_sentence = ''

        for segment_file in segment_files:
            diarization = pipeline(segment_file, min_speakers=2, max_speakers=2)
            segment_transcriptions = []
            prev_speaker = None
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                audio_segment_path = f"_segment_{turn.start}_{turn.end}.wav"
                extract_segment(segment_file, turn.start, turn.end, audio_segment_path)
                segment_transcription = self.model.transcribe(audio_segment_path, language='en')
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

        speaker_transcription = self.model.transcribe(file_path, language='en')['text']
        transcription = get_sentence_case(speaker_transcription.strip())
        print(transcription)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = whisper.load_model(m)
