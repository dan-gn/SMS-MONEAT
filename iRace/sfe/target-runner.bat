@echo off

set CONDAPATH=C:\Users\user\anaconda3
set ENVNAME=moneat

if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

set parameters=%*
python target-runner.py  %parameters%