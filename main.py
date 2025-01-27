import os

from dotenv import load_dotenv

load_dotenv()

import argparse
import time
import logging
from logging.handlers import TimedRotatingFileHandler
import traceback
import sys

from core.transcribe import Transcribe
from utils.logger_config import logger


parser = argparse.ArgumentParser(description="Transcribe audio files with specified settings.")
parser.add_argument('--model', type=str, default='medium', help='Model type to use for transcription (e.g., large, medium, small).')
parser.add_argument('--sleep', type=int, default=5, help='Sleep duration in minutes between transcriptions.')

args = parser.parse_args()


if os.environ.get('environment') == 'test':
    print('here')
    from core.tests import Transcribe
    t = Transcribe(model=args.model)
    t.transcribe()
    sys.exit()

#  Please run t.transcribe to auto perform all necessary functions.
#  Using this function will prevent from having to manually call other functions.

t = Transcribe(model=args.model)
while True:
    sleep_time_in_minutes = 5
    try:
        t.transcribe()
    except Exception as e:
        logger.error(f'{traceback.format_exc()}')
    logger.debug('-' * 50)
    logger.debug(f'Sleeping for {sleep_time_in_minutes} minutes')
    logger.debug('-' * 50)
    time.sleep(60 * sleep_time_in_minutes)
