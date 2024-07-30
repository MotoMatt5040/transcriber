# Installing Dependencies

### Python 3.12.4

This script is designed to run on Windows, but can be modified to run on other systems
 
Install chocolatey at https://chocolatey.org/install#individual

Make sure to use the individual installation and copy the string text it provides and paste into an Administrative Powershell.

Here is the link provided for you at the time of writing this program 6/25/2024
```bash 
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

If you run into errors you may want to run the following commands before attempting the above command again.
```bash  
$chocolateyPath = "C:\ProgramData\chocolatey\bin"
$env:Path += ";$chocolateyPath"
```  

Next install the FFMPEG package using the following command
```bash
choco install ffmpeg
```
##

Now, create a python virtual environment using 

## virtualenv: (Python Default)
```bash
python -m venv env
.\venv\Scripts\activate
```

## Anaconda: (Check Issues for installation)
```bash
conda create -n env python=3.12.4
conda activate env
```
##
PyTorch can now be installed, browse to https://pytorch.org/get-started/locally/ for your specific needs or follow the setup below.

Here you will select your system options, only use "CUDA" if you have an NVIDIA GPU and have installed the CUDA Toolkit (Check Issues below).

# Pre-Written Installation Commands (Or skip to Custom)
You may choose ONE of the 4 pre-written commands you can use to install packages. They have been split into 2 separate categories for GPU and CPU installations. You may also choose NOT to use the commands written below, if that is the case you may skip to the Custom section.

## GPU package

### pip:
```bash
pip3 install torch torchvision torchaudio openai-whisper pyodbc sqlalchemy python-dotenv --index-url https://download.pytorch.org/whl/cu124
```
OR
### conda:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia pyodbc sqlalchemy
```
##

## CPU package

### pip:
```bash
pip3 install torch torchvision torchaudio openai-whisper pyodbc sqlalchemy python-dotenv --index-url https://download.pytorch.org/whl/cpu
```
OR
### conda:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch pyodbc sqlalchemy
```
##
# Custom (If you did not run one of the above commands)
Install the following dependencies.
## pip:
```bash
pip install pyodbc sqlalchemy python-dotenv openai-whisper
```

## conda:
```bash
conda install pyodbc sqlalchemy
```
##

Finally, run the following command to install the remaining dependencies in conda. This step is skipped when using virtualenv and pip
## conda:
I did have to use pip inside conda, while not ideal, it is a workaround
```bash
pip install openai-whisper python-dotenv
```
##

You now have all dependencies installed and can run the following script

Files can be transcribed through the command prompt or terminal using the command ```whisper <filename>```

Multiple files can be done using ```whisper <filename1> <filename2> <filename3>```

If there are spaces in your filename, use quotes around the file name

You can use ```--model``` to specify the model you want to use

# Issues (Windows Only)

Anaconda can be installed by downloading the installer from https://www.anaconda.com/products/individual and running the installer. If you run into issues, you may need to run the installer as an administrator.

If you are using an NVIDIA GPU that is not a datacenter GPU, and if you don't need to run very intensive tasks, you may install GeForce Experience. This will install the necessary drivers for your GPU. If you are using a datacenter card, you may need to install the drivers manually from https://www.nvidia.com/download/index.aspx. If you need to run more intensive tasks, you may want to opt for the CUDA Toolkit installation instead.

If you are installing on Windows Server 2022 and run into wlanapi.dll is missing, follow the instructions here https://www.nvidia.com/en-us/geforce/forums/geforce-experience/14/347710/can-not-find-wlanapi-dll-file-when-geforec-experie/

Verify that your GPU is CUDA compatible at https://developer.nvidia.com/cuda-gpus

You will need to have Visual Studio installed to use the CUDA Toolkit. You can download Visual Studio at https://visualstudio.microsoft.com/vs/

If you are using an NVIDIA GPU, you may need to install the CUDA Toolkit at https://developer.nvidia.com/cuda-toolkit-archive

Make sure you have the correct version of the CUDA Toolkit installed for your GPU and the version of PyTorch you are using (12.4 is what was used in this setup).

It is HIGHLY recommended to use Anaconda for package management if you are using a GPU. It is a much easier way to manage your packages and environments. Most of the issues you may run into can be solved by simply using Anaconda.

If you are having issues with sqlalchemy connecting to a server, ensure you have the proper ODBC driver installed. Driver 18 is the current most up-to-date version: https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver16

If you haven't already, you may need to install the C++ redistributable package for Visual Studio 2019: https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0


