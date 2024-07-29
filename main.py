from core.transcribe import Transcribe
import time
from datetime import datetime

#  Please run self.transcribe to auto perform all necessary functions.
#  Using this function will prevent from having to manually call other functions.
t = Transcribe()
while True:
    current_hour = int(datetime.now().strftime("%H"))
    if not current_hour < 2 or not current_hour > 15:
        print("Sleeping for 30 minutes")
        time.sleep(60 * 30)
        continue
    t.transcribe()
    print('\n\n' + '-' * 50)
    print('Sleeping for 30 minutes')
    print('-' * 50 + '\n\n')
    time.sleep(60*30)


