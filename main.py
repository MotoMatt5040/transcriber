from dotenv import load_dotenv
load_dotenv()

import os
import whisper

cwd = os.getcwd()
audio_path = os.path.join(cwd, "audio")
audio_file_list = os.listdir(audio_path)
error_report = []
model = whisper.load_model("base")

for file_name in audio_file_list:
    if not file_name.endswith(".wav"):
        continue

    file_path = os.path.join(audio_path, file_name)

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

if error_report:
    with open("error_report.txt", "w") as f:
        f.write("\n".join(error_report))
