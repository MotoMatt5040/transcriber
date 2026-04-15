import torch
import os
import whisper
import time

from utils.models import ProjectTranscriptionManager, session
from pyannote.audio import Pipeline

from utils.logger_config import logger
from core.audio_utils import (
    estimate_time, clean_text, extract_segment,
    preprocess_audio, get_audio_length, get_sentence_case,
    cleanup_temp_files,
)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=os.environ['HUGGING_FACE_TOKEN'])

# TODO Test with Q17OE 12948 12751

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


ptm = ProjectTranscriptionManager(session)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', file_path='', print_end=""):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    est = estimate_time(total, iteration, cuda_device_name)
    print(f'\r{prefix} |{bar}| {est}s - {iteration}/{total} - {percent}% {suffix} - {file_path}', end=print_end)
    logger.info(f"{prefix} {iteration}/{total} - {percent}% - ~{est}s remaining - {file_path}")
    if iteration == total:
        print()


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


COMMIT_BATCH_SIZE = 20


class Transcribe:
    """
    Transcribe class to transcribe audio files.

    Default load_model is medium if no model is specified
    """
    def __init__(self, model: str):
        self._model = whisper.load_model(model, device="cuda")
        self._model.to(torch.device('cuda'))

    def transcribe(self):
        res = ptm.projects_to_transcribe()
        if not res:
            logger.debug("Projects to transcribe is empty")
            return

        text_removal, result = res
        amount = len(result)

        estimated_time = estimate_time(amount, cuda_device_name=cuda_device_name)

        if amount == 0:
            return
        logger.info(f'Total amount of records to transcribe: {amount}')
        logger.info(f'Estimated time to complete: {estimated_time}s')
        time.sleep(0.1)
        start = time.perf_counter()
        print_progress_bar(0, amount, prefix='Progress:', suffix='Complete', length=50)
        pending_commits = 0
        if result is not None:
            for i, item in enumerate(result):
                if 'Proof' in item.RecStrID:
                    logger.debug('proof')
                    continue

                if item.Transcription is not None:
                    logger.debug('transcription')
                    continue

                if not item.ProjectID:
                    logger.debug('no project id')
                    continue

                file_path = f"{os.environ['WAV_PATH_BEGIN']}{item.PCMHome}{os.environ['WAV_PATH_END']}{item.ProjectID}PCM/{item.Question}_{item.SurveyID}.wav"
                logger.debug(f"Processing: {file_path}")

                if not os.path.exists(file_path):
                    item.Transcription = ''
                    logger.debug(f"File does not exist for: {file_path}")
                    continue

                audio_length = get_audio_length(file_path)

                if audio_length < 10:
                    item.Transcription = ''
                    logger.warning(f"Audio file is too short: {file_path}")
                    continue

                if audio_length > 99600:
                    item.Transcription = ''
                    logger.warning(f"Audio file is too long: {file_path}")
                    continue

                print_progress_bar(i + 1, amount, prefix='Progress:', suffix='Complete', length=50, file_path=file_path)

                temp_files = []
                try:
                    processed_file = preprocess_audio(file_path)
                    temp_files.append(processed_file)

                    diarization = pipeline(processed_file, min_speakers=2, max_speakers=2)

                    speaker_transcription = ''
                    current_sentence = ''
                    all_transcriptions = []

                    interviewer = None
                    respondent = None
                    prev_speaker = None
                    # TODO fix repeats using 12926 Q20_1OE 0000044033
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        audio_segment_path = f"_segment_{turn.start}_{turn.end}.wav"
                        extract_segment(processed_file, turn.start, turn.end, audio_segment_path)

                        segment_transcription = self.model.transcribe(audio_segment_path, language='en')
                        text = segment_transcription['text']
                        os.remove(audio_segment_path)

                        if not prev_speaker or prev_speaker != speaker:
                            current_sentence = f"{speaker}: {text}"
                            prev_speaker = speaker
                        else:
                            current_sentence += text

                        all_transcriptions.append(text)

                        if not interviewer:
                            th = TextHandler(text, text_removal[item.Question])
                            match = sum(1 for t in th.question if t in current_sentence)
                            p_match = match * 100 / len(th.question)
                            if p_match > 75:
                                interviewer = speaker
                                respondent = "SPEAKER_01" if interviewer == "SPEAKER_00" else "SPEAKER_00"
                            continue

                        if speaker == respondent:
                            speaker_transcription += text

                    if not interviewer:
                        speaker_transcription = ' '.join(all_transcriptions)

                    item.Transcription = get_sentence_case(speaker_transcription.strip())
                    logger.info(f"Transcribed: {item.ProjectID}/{item.Question}/{item.SurveyID}")
                    pending_commits += 1

                    if pending_commits >= COMMIT_BATCH_SIZE:
                        try:
                            session.commit()
                            pending_commits = 0
                        except Exception as e:
                            logger.error(f"Batch commit failed: {e}")
                            session.rollback()
                finally:
                    cleanup_temp_files(*temp_files)

        # Commit any remaining records
        try:
            session.commit()
        except Exception as e:
            logger.error(f"Final commit failed: {e}")
            session.rollback()

        end = time.perf_counter()
        elapsed = round(end - start)
        print()
        print(f'Transcriptions completed in {elapsed}s')
        logger.info(f'Transcriptions completed: {amount} records in {elapsed}s')

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = whisper.load_model(m)
