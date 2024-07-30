import torch
from core.transcribe import Transcribe
import time


print("CUDA is available:", torch.cuda.is_available())
print("CUDA devices available:", torch.cuda.device_count())
print("Current CUDA device", torch.cuda.current_device())
print("Current CUDA device address", torch.cuda.device(0))
print("Current CUDA device name:", torch.cuda.get_device_name(0))


#  Please run t.transcribe to auto perform all necessary functions.
#  Using this function will prevent from having to manually call other functions.
t = Transcribe()
while True:
    sleep_time_in_minutes = 5
    t.transcribe()
    print('\n\n' + '-' * 50)
    print(f'Sleeping for {sleep_time_in_minutes} minutes')
    print('-' * 50 + '\n\n')
    time.sleep(60 * sleep_time_in_minutes)
