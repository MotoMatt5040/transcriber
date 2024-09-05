import torch
import os
import whisper
import time
import re

from utils.models import ProjectTranscriptionManager, session
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pyannote.audio import Pipeline

from utils.logger_config import logger

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ['hugging_face_token'])

cuda_is_available = torch.cuda.is_available()
cuda_device_name = None
if cuda_is_available:
    logger.info(f"CUDA is available {cuda_is_available}")
    logger.info(f"CUDA devices available: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device {torch.cuda.current_device()}")
    logger.info(f"Current CUDA device address {torch.cuda.device(0)}")
    cuda_device_name = torch.cuda.get_device_name(0)
    logger.info(f"Current CUDA device name: {cuda_device_name}")

    torch.set_default_device('cuda')
    pipeline.to(torch.device('cuda'))

else:
    logger.warning("CUDA is not available, using CPU instead")
    torch.set_default_device('cpu')
    pipeline.to(torch.device('cpu'))


ptm = ProjectTranscriptionManager(session)
model = whisper.load_model("medium")


def print_red(*args):
    message = ' '.join(map(str, args))
    print(f"\033[91m{message}\033[0m")


def print_light_blue(*args):
    message = ' '.join(map(str, args))
    print(f"\033[94m{message}\033[0m")


def print_light_green(*args):
    # Convert all arguments to string and concatenate them
    message = ' '.join(map(str, args))
    print(f"\033[92m{message}\033[0m")


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end=""):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {estimate_time(total, iteration)}s - {iteration}/{total} - {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


def estimate_time(amount: int, iteration: int = 0) -> int:
    match cuda_device_name:
        case "NVIDIA GeForce RTX 4090":
            return round((amount - iteration) * 2)
        case "Tesla T4":
            return round((amount - iteration) * 8)
        case _:
            return round((amount - iteration) * 16)


def clean_text(text: str) -> list:
    """
    Clean text by removing commas, converting to lowercase, removing font tags, and splitting the text.
    :param text:
    :return: list of words in text
    """
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
    combined_audio = AudioSegment.silent(duration=0)  # Start with an empty silent segment
    for segment in trimmed_audio:
        combined_audio += segment + AudioSegment.silent(duration=500)  # Add a bit of silence between segments
    combined_audio.export(output_file, format='wav')


def preprocess_audio(file_path):
    temp_file = f"_processed.wav"

    # Convert to WAV format if needed
    convert_audio_format(file_path, temp_file)

    # Normalize volume
    normalized_file = f"{temp_file}_normalized.wav"
    normalize_audio(temp_file, normalized_file)

    # Resample audio to 16kHz
    resampled_file = f"{normalized_file}_resampled.wav"
    resample_audio(normalized_file, resampled_file)

    # Trim silence
    trimmed_file = f"{resampled_file}_trimmed.wav"
    trim_silence(resampled_file, trimmed_file)

    return trimmed_file


def get_audio_length(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000  # Duration in seconds
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
    """
    Transcribe class to transcribe audio files.

    Default load_model is medium if no model is specified
    """
    def __init__(self, model: str = "medium"):
        self.model = model
        self.transcription_dict = {}
        self.transcription_errors = []

    def transcribe(self):
        # self.transcription_json()
        res = ptm.projects_to_transcribe()
        if not res:
            logger.debug("Projects to transcribe is empty")
            return

        text_removal, result = res
        amount = len(result)

        estimated_time = estimate_time(amount)

        if amount == 0:
            return
        logger.info(f'Total amount of records to transcribe: {amount}')
        logger.info(f'Estimated time to complete: {estimated_time}s')
        time.sleep(0.1)
        start = time.perf_counter()
        print_progress_bar(0, amount, prefix='Progress:', suffix='Complete', length=50)
        if result is not None:
            for i, item in enumerate(result):
                if 'Proof' in item.RecStrID:
                    continue

                if item.Transcription is not None:
                    continue

                if not item.ProjectID:
                    continue

                file_path = None
                file_name = f'{item.Question}_{item.SurveyID}.wav'
                for p_val in range(1, 3):
                    base_path = f"{os.environ['wav_path_begin']}{p_val}{os.environ['wav_path_end']}/{item.ProjectID}PCM"
                    if not os.path.exists(base_path):
                        continue
                    file_path = f'{base_path}/{file_name}'
                    if not os.path.exists(file_path):
                        file_path = None
                        continue

                if not file_path:
                    logger.debug(f"File does not exist for {file_name}")
                    continue

                if get_audio_length(file_path) < 10:
                    item.Transcription = ''
                    logger.warning(f"Audio file is too short: {file_path}")
                    continue

                processed_file = preprocess_audio(file_path)

                # Segment the audio based on silence if needed
                segment_files = segment_audio_by_silence(processed_file)

                # Transcribe each segment
                full_transcription = ""
                temp = []
                speaker_transcription = ''
                current_sentence = ''

                interviewer = None

                for segment_file in segment_files:
                    # Perform diarization again on the segment if necessary
                    diarization = pipeline(segment_file, min_speakers=2, max_speakers=2)
                    segment_transcriptions = []
                    prev_speaker = None
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        audio_segment_path = f"_segment_{turn.start}_{turn.end}.wav"
                        extract_segment(segment_file, turn.start, turn.end, audio_segment_path)

                        # Transcribe the audio segment
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

                        # Delete the temporary segment file after transcription
                        os.remove(audio_segment_path)

                        th = TextHandler(segment_transcription['text'], text_removal[item.Question])
                        match = 0

                        if not interviewer:
                            for t in th.question:
                                if t in current_sentence:
                                    match += 1
                            p_match = match * 100 / len(th.question)
                            if p_match > 75:
                                interviewer = speaker
                            continue

                        if interviewer == "SPEAKER_00":
                            respondent = "SPEAKER_01"
                        else:
                            respondent = "SPEAKER_00"

                        if speaker == respondent:
                            speaker_transcription += segment_transcription['text']

                    full_transcription += " ".join(segment_transcriptions) + " "
                    temp.append(segment_transcriptions)

                if not interviewer:
                    logger.warning(f"Interviewer not found: {item.ProjectID} - {item.Question} - {item.SurveyID} - {file_path}")
                    speaker_transcription = model.transcribe(file_path, language='en')['text']

                item.Transcription = get_sentence_case(speaker_transcription.strip())
                session.commit()
                print_progress_bar(i + 1, amount, prefix='Progress:', suffix='Complete', length=50)
        end = time.perf_counter()
        logger.info(f'Transcriptions completed in {round(end - start)}s')

        if self.transcription_errors:
            logger.error(f'Transcription errors were found. Records have been recorded in the transcription_errors.log file.')
            for err in self.transcription_errors:
                logger.error("    ", err)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = whisper.load_model(m)

class TextHandler:

    def __init__(self, text, text_removal):
        self.text = text
        self.question = text_removal['text'].lower()
        self.probe = text_removal['probe'].lower()

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, t: str):
        t = t.strip()
        t = t.replace(',', '')
        self._text = t

    @property
    def question(self) -> list:
        return self._question

    @question.setter
    def question(self, q: str):
        self._question = clean_text(q)

    @property
    def probe(self) -> list:
        return self._probe

    @probe.setter
    def probe(self, p: str):
        self._probe = clean_text(p)
