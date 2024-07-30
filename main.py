import torch
print("CUDA is available:", torch.cuda.is_available())
print("CUDA devices available:", torch.cuda.device_count())
print("Current CUDA device", torch.cuda.current_device())
print("Current CUDA device address", torch.cuda.device(0))
print("Current CUDA device name:", torch.cuda.get_device_name(0))


from core.transcribe import Transcribe
import time
# from datetime import datetime


#  Please run self.transcribe to auto perform all necessary functions.
#  Using this function will prevent from having to manually call other functions.
t = Transcribe()
while True:
    sleep_time_in_minutes = 5
    # current_time = datetime.now().strftime("%H:%M:%S")
    # start_time = "16:00:00"
    # end_time = "23:59:00"
    # if start_time <= current_time or current_time <= end_time:
    #     print(start_time, current_time, end_time)
    #     print("Sleeping for 30 minutes")
    #     time.sleep(60 * 30)
    #     continue
    t.transcribe()
    print('\n\n' + '-' * 50)
    print(f'Sleeping for {sleep_time_in_minutes} minutes')
    print('-' * 50 + '\n\n')
    time.sleep(60 * sleep_time_in_minutes)


