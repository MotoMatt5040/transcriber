import os
import whisper

cwd = os.getcwd()
dir = os.path.join(cwd, "audio")

audio_file_list = os.listdir(dir)

error_report = []
file_name = "Q4AOE_0000302814.wav"
for file_name in audio_file_list:
    if not file_name.endswith(".wav"):
        continue
    file_path = os.path.join(dir, file_name)
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        write_path = f"results/{file_name.replace('.wav', '')}.txt"
        with open(write_path, "w", encoding='utf-8') as f:
            try:
                f.write(result['text'])
            except:
                print(f"{file_name}: Failed to write to file")
    except Exception as e:
        print(f"{file_name}: Failed to load model\n    {e}")
        error_report.append(f"{file_name}: Failed to load model\n    {e}")

if error_report:
    with open("error_report.txt", "w") as f:
        f.write("\n".join(error_report))
