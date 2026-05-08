set env=%USERPROFILE%\mambaforge\envs\ls-sam2
call %USERPROFILE%\mambaforge\condabin\activate.bat %env%
call activate ls-sam2
cd C:\ITR\label-studio-ml-backend\label_studio_ml\examples
label-studio-ml start ./segment_anything_2_image -p 9090