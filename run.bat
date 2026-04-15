@ECHO OFF
TITLE Execute python script on anaconda environment
ECHO Please Wait...

:: Set working directory to where this script lives
cd /d "%~dp0"

:: Section 1: Activate the environment.
ECHO ============================
ECHO Conda Activate
ECHO ============================
@CALL "%LOCALAPPDATA%\anaconda3\Scripts\activate.bat" transcription

:: Section 2: Execute python script.
ECHO ============================
ECHO Python main.py
ECHO ============================
python main.py --model large --sleep 5

ECHO ============================
ECHO End
ECHO ============================

PAUSE
