@echo off
echo Installing dependencies in virtual environment...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch (CPU version first for compatibility)
echo Installing PyTorch...
python -m pip install torch torchvision torchaudio

REM Install core ML libraries
echo Installing transformers and related libraries...
python -m pip install transformers accelerate tokenizers

REM Install networking libraries
echo Installing networking libraries...
python -m pip install fastapi uvicorn websockets httpx pydantic

REM Install utility libraries
echo Installing utility libraries...
python -m pip install psutil pytest datasets evaluate

REM Install data science libraries
echo Installing data science libraries...
python -m pip install tqdm numpy pandas matplotlib seaborn

echo.
echo Installation completed!
echo.
echo To test the installation, run:
echo   python -c "import torch; import transformers; print('Installation successful!')"
echo.
pause
