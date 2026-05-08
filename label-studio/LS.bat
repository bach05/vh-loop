set env=%USERPROFILE%\mambaforge\envs\ls-ui
call %USERPROFILE%\mambaforge\condabin\activate.bat %env%
call activate ls-ui
label-studio start
