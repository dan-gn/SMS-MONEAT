@echo off

set CONDAPATH=C:\Users\23252359\AppData\Local\anaconda3
set ENVNAME=elbueno

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

set parameters=%*
python target-runner.py  %parameters%