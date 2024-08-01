import argparse
import time

from dotenv import load_dotenv
from core.transcribe import Transcribe

load_dotenv()


parser = argparse.ArgumentParser(description="Transcribe audio files with specified settings.")
parser.add_argument('--model', type=str, default='large', help='Model type to use for transcription (e.g., large, medium, small).')
parser.add_argument('--sleep', type=int, default=5, help='Sleep duration in minutes between transcriptions.')

args = parser.parse_args()


#  Please run t.transcribe to auto perform all necessary functions.
#  Using this function will prevent from having to manually call other functions.
t = Transcribe(model=args.model)
while True:
    sleep_time_in_minutes = 5
    t.transcribe()
    print('\n\n' + '-' * 50)
    print(f'Sleeping for {sleep_time_in_minutes} minutes')
    print('-' * 50 + '\n\n')
    time.sleep(60 * sleep_time_in_minutes)
