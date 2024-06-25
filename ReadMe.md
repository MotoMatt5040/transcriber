# Installing dependencies
This script is designed to run on Windows, but can be modified to run on other systems
 
Install chocolatey at https://chocolatey.org/install#individual

Make sure to use the individual installation

Here is the link provided for you at the time of writing this program 6/25/2024
```bash 
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```    

Copy the string text it provides and paste into an Administrative Powershell.
If you run into errors you may want to run the follow commands first
```bash  
$chocolateyPath = "C:\ProgramData\chocolatey\bin"
$env:Path += ";$chocolateyPath"
```  

Next install the FFMPEG package using the following command
```bash
choco install ffmpeg
```

Now, create a python virtual environment using
```bash
python -m venv venv
.\venv\Scripts\activate
```

PyTorch can now be installed, browse to https://pytorch.org/get-started/locally/

Here you will select your system options, only use "CUDA" if you have an NVIDIA

Here is the current cpu package
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Here is the current GPU package
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Run one of the above commands inside your virtual environment terminal

Finally, run the following command
```bash
pip install openai-whisper
```

You how have all dependencies installed and can run the following script

Files can be transcribed through the command prompt or terminal using the command ```whisper <filename>```

Multiple files can be done using ```whisper <filename1> <filename2> <filename3>```

If there are spaces in your filename, use quotes around the file name

You can use ```--model``` to specify the model you want to use
