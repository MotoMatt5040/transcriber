from core.transcribe import Transcribe
import time

from dotenv import load_dotenv

load_dotenv()


#  Please run t.transcribe to auto perform all necessary functions.
#  Using this function will prevent from having to manually call other functions.
t = Transcribe(load_model='large')
while True:
    sleep_time_in_minutes = 5
    t.transcribe()
    print('\n\n' + '-' * 50)
    print(f'Sleeping for {sleep_time_in_minutes} minutes')
    print('-' * 50 + '\n\n')
    time.sleep(60 * sleep_time_in_minutes)
