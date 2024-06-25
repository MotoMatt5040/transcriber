# Installing dependencies
# This script is designed to run on Windows, but can be modified to run on other systems
# Install chocolatey at https://chocolatey.org/install#individual
# Make sure to use the individual installation
# Here is the link provided for you at the time of writing this program 6/25/2024
# Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Copy the string text it provides and paste into an Administrative Powershell.
# If you run into errors you may want to run the follow commands first
# $chocolateyPath = "C:\ProgramData\chocolatey\bin"
# $env:Path += ";$chocolateyPath"

# Next install the FFMPEG package using the following command
# choco install ffmpeg

# Now, create a python virtual environment using
# python -m venv venv
# .\venv\Scripts\activate

# PyTorch can now be installed, browse to https://pytorch.org/get-started/locally/
# Here you will select your system options, only use "CUDA" if you have an NVIDIA
# Here is the current cpu package
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Here is the current GPU package
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Run this command inside of your virtual environment terminal

# Finally, run the following command
# pip install openai-whisper

# You how have all dependencies installed and can run the following script

# Files can be transcribed through the command prompt or terminal
# To do so, use the commend "whisper <filename>"
# Multiple files can be done using "whisper <filename1> <filename2> <filename3>"
# If there are spaces in your filename, use quotes around the file name
# You can use --model to specify the model you want to use
# The rest of this file will be a script created to run multiple files at one time
import os
import whisper

cwd = os.getcwd()
dir = os.path.join(cwd, "audio")

audio_file_list = os.listdir(dir)

error_report = []
results = {}
for file_name in audio_file_list:
    if not file_name.endswith(".wav"):
        continue
    file_path = os.path.join(dir, file_name)
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        results[file_name] = result['text']
    except Exception as e:
        error_report.append(f"{file_name}: Failed to load model\n    {e}")

if error_report:
    with open("error_report.txt", "w") as f:
        f.write("\n".join(error_report))

for k, v in results.items():
    with open(f"results/{k.replace('.wav', '')}.txt", "w") as f:
        f.write(v)
