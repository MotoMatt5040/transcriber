from dotenv import load_dotenv
load_dotenv()

import os
import whisper
import time


cwd = os.getcwd()
audio_path = os.path.join(cwd, "audio")
print(audio_path)
audio_file_list = os.listdir(audio_path)
error_report = []
model = whisper.load_model("medium")
amount = len(audio_file_list)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = ""):
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
    print(f'\r{prefix} |{bar}| {round((total - iteration) * 2)}s - {iteration}/{total} - {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def transcribe(file_path, file_name):
    try:

        result = model.transcribe(file_path)
        write_path = f"results/{file_name.replace('.wav', '')}.txt"

        with open(write_path, "w", encoding='utf-8') as f:
            try:
                f.write(result['text'])
            except Exception as e:
                print(f"{file_name}: Failed to write to file\n    {e}")
    except Exception as e:
        print(f"{file_name}: Failed to load model\n    {e}")
        error_report.append(f"{file_name}: Failed to load model\n    {e}")


def main():
    for i, file_name in enumerate(audio_file_list):

        if not file_name.endswith(".wav"):
            continue

        file_path = os.path.join(audio_path, file_name)

        transcribe(file_path, file_name)

        printProgressBar(i + 1, amount, prefix='Progress:', suffix='Complete', length=50)


printProgressBar(0, amount, prefix='Progress:', suffix='Complete', length=50)
start = time.perf_counter()
# asyncio.run(main())
main()

if error_report:
    with open("error_report.txt", "w") as f:
        f.write("\n".join(error_report))

end = time.perf_counter()
print()
print(f"Time: {end-start}")
