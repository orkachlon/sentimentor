@echo off
set review=%1
cd "C:\ProgramData\Anaconda3\Scripts"
CALL "C:\ProgramData\Anaconda3\condabin\conda.bat" activate ml-as-tool-project-1
cd "C:\Users\Or Kachlon\Documents\ml-as-tool\interactive-ml-project-1\dev"
CALL python nlp_module.py %review%
CALL "C:\ProgramData\Anaconda3\condabin\conda.bat" deactivate
