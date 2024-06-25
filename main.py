# Files can be transcribed through the command prompt or terminal
# To do so, use the commend "whisper <filename>"
# Multiple files can be done using "whisper <filename1> <filename2> <filename3>"
# If there are spaces in your filename, use quotes around the file name
# You can use --model to specify the model you want to use
# The rest of this file will be a script created to run multiple files at one time
import os
import whisper

dir = os.getcwd()

print(dir)

# model = whisper.load_model("base")
# result = model.transcribe("test.wav")