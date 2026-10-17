@echo off
chcp 65001 >nul
echo ===== Creando LSB_Reconocimiento.exe =====
where python >nul 2>nul || (echo No se encontro Python. Instala Python 3.10-3.12 desde python.org y marca "Add to PATH". & pause & exit /b 1)
python -m venv venv_exe || (pause & exit /b 1)
call venv_exe\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo Fallo la instalacion. & pause & exit /b 1)
pyinstaller --noconfirm --clean --windowed --name LSB_Reconocimiento --add-data "model.onnx;." --add-data "labels.txt;." lsb_app.py || (echo Fallo PyInstaller. & pause & exit /b 1)
echo.
echo LISTO. El programa esta en:  dist\LSB_Reconocimiento\LSB_Reconocimiento.exe
echo Para llevarlo a otra PC copia la carpeta completa dist\LSB_Reconocimiento
pause
