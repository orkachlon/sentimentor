@echo off
:: path to conda /Scripts/ dir
set conda_path=%1
:: path to sketch directory
set working_dir=%2
:: the review to process
set review=%3

:: activate env
cd %conda_path%
CALL activate ml-as-tool-project-1
:: analyze review
cd %working_dir%
CALL python nlp_module.py %review%
:: deactivate env
CALL conda deactivate