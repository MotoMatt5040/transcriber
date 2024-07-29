from core.transcribe import Transcribe
import time
from datetime import datetime

#  Please run self.transcribe to auto perform all necessary functions.
#  Using this function will prevent from having to manually call other functions.
t = Transcribe()
while True:
    sleep_time_in_minutes = 5
    current_hour = int(datetime.now().strftime("%H"))
    if not current_hour < 2 or not current_hour > 15:
        print("Sleeping for 30 minutes")
        time.sleep(60 * 30)
        continue
    t.transcribe()
    print('\n\n' + '-' * 50)
    print(f'Sleeping for {sleep_time_in_minutes} minutes')
    print('-' * 50 + '\n\n')
    time.sleep(60 * sleep_time_in_minutes)


