@echo off
:: path to sketch directory
set nlp_dir=%1
:: the review to process
set review=%2

:: activate env
CALL conda activate ml-as-tool-project-1
:: analyze review
cd %nlp_dir%
CALL python nlp_module.py %review%
:: deactivate env
CALL conda deactivate