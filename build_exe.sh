#!/bin/bash
### Window requires pyinstaller

rm -rf ../dist
rm -rf ../build

conda activate py37_headpose

python.exe c:/Users/83538/.conda/envs/py37_headpose/Scripts/pyinstaller.exe labelImg.py -F
python.exe c:/Users/83538/.conda/envs/py37_headpose/Scripts/pyinstaller.exe labelImg.py -F -w
python.exe c:/Users/83538/.conda/envs/py37_headpose/Scripts/pyinstaller.exe --onefile labelImg_v6.spec -F
python.exe c:/Users/83538/.conda/envs/py37_headpose/Scripts/pyinstaller.exe --onefile labelImg_v6.spec -F -w



# fix bugs when running pyinstaller
C:\Users\83538\.conda\envs\py37_headpose\Lib\site-packages\mayavi\core\utils.py
change
import vtk
into
import vtkmodules.all as vtk

C:\Users\83538\.conda\envs\py37_headpose\Lib\site-packages\tvtk\tools\ivtk.py
change
from pyface.api import FileDialog, GUI, OK, PythonShell
into
from pyface.api import FileDialog, GUI, OK
from pyface.python_shell import PythonShell

https://github.com/tzutalin/labelImg/issues/62
copy folder
C:\Users\83538\.conda\envs\py37_headpose\Lib\site-packages\PyQt4\plugins\imageformats\qjpeg4.dll
into the same folder <imageformats/qjpeg4.dll> with .exe file to support loading .jpg
