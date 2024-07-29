import logging

from dotenv import load_dotenv
load_dotenv()

import os
import whisper
import time
import json
import asyncio

from datetime import date, timedelta

from utils.models import ProjectTranscriptionManager, session


ptm = ProjectTranscriptionManager(session)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
logger.propagate = True  # pass logs to root logger
model = whisper.load_model("medium")


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
    print(f'\r{prefix} |{bar}| {round((total - iteration) * 2)}s - {iteration}/{total} - {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()


class Transcribe:
    """
    Transcribe class to transcribe audio files.

    Default load_model is medium if no model is specified
    """
    def __init__(self, load_model: str = "medium"):
        self.model = whisper.load_model(load_model)
        self.transcription_dict = {}
        self.transcription_errors = []

    def transcription_json(self):
        for project in ptm.get_active_projects():
            self.transcription_dict[project.ProjectID] = {}
            self.transcription_dict[project.ProjectID]['records'] = []
            if project.ProjectID.upper().endswith("C"):
                wav_path = rf'{os.environ['cell_wav']}{project.ProjectID}PCM'
                self.transcription_dict[project.ProjectID]['wav_path'] = wav_path
                self.transcription_dict[project.ProjectID]['wav'] = os.listdir(wav_path)
            else:
                wav_path = rf'{os.environ['landline_wav']}{project.ProjectID}PCM'
                self.transcription_dict[project.ProjectID]['wav_path'] = wav_path
                self.transcription_dict[project.ProjectID]['wav'] = os.listdir(wav_path)

    def transcribe(self):
        self.transcription_json()
        text_removal, result = ptm.projects_to_transcribe()
        print(text_removal)
        amount = len(result)
        if amount == 0:
            return
        print(f'Total amount of records to transcribe: {amount}')
        print(f'Estimated time to complete: {round(amount * 2)}s')
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

                if not self.transcription_dict.get(item.ProjectID):
                    self.transcription_errors.append(item.ProjectID)

                file_path = f'{item.Question}_{item.SurveyID}.wav'
                transcription = model.transcribe(rf'{self.transcription_dict[item.ProjectID]['wav_path']}\{file_path}')
                text = transcription['text']
                match = 0

                first_q = text.split('?')[0]
                # print('-'*10)
                # print(first_q)

                for t in text_removal[item.Question]['text']:
                    if t in first_q:
                        match += 1
                # print(match)
                if match * 100 / len(text_removal[item.Question]['text']) > 75:
                    # print("75")
                    text = text[len(first_q) + 2:]
                # text = text.replace(text_removal[item.Question]['text'], '').replace(text_removal[item.Question]['probe'], '')
                item.Transcription = text
                # print(text)
                print_progress_bar(i + 1, amount, prefix='Progress:', suffix='Complete', length=50)

        session.commit()
        end = time.perf_counter()
        print(f'Transcription completed in {round(end - start)}s')

        if self.transcription_errors:
            logger.warning(f'Transcription errors were found. Records have been recorded in the transcription_errors.log file.')
            for err in self.transcription_errors:
                logger.warning("    ", err)

