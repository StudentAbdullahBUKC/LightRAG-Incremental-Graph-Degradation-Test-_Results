@echo off
REM LightRAG Microstudy — Environment Setup (Windows)
REM ==================================================
REM Run this ONCE before running the experiment.
REM Usage: experiments\setup_environment.bat

echo ==============================================
echo   LightRAG Microstudy — Environment Setup
echo ==============================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

echo Project directory: %PROJECT_DIR%
echo.

REM -----------------------------------------------------------
REM 1. Check Ollama
REM -----------------------------------------------------------
echo [Step 1/6] Checking Ollama installation...
where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   √ Ollama is installed
) else (
    echo   X Ollama not found.
    echo     Please download and install from: https://ollama.com/download/windows
    echo     Then re-run this script.
    pause
    exit /b 1
)

REM -----------------------------------------------------------
REM 2. Ensure Ollama is running
REM -----------------------------------------------------------
echo.
echo [Step 2/6] Ensuring Ollama server is running...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   √ Ollama server is responding
) else (
    echo   Starting Ollama server...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo   √ Ollama server started
    ) else (
        echo   X FATAL: Cannot reach Ollama server
        echo     Try running 'ollama serve' manually in another terminal.
        pause
        exit /b 1
    )
)

REM -----------------------------------------------------------
REM 3. Pull required models
REM -----------------------------------------------------------
echo.
echo [Step 3/6] Pulling required models...
echo   Pulling llama3.1:7b (~4.7GB)...
ollama pull llama3.1:7b
echo   √ llama3.1:7b ready

echo   Pulling nomic-embed-text:latest (~274MB)...
ollama pull nomic-embed-text:latest
echo   √ nomic-embed-text ready

REM -----------------------------------------------------------
REM 4. Install Python dependencies
REM -----------------------------------------------------------
echo.
echo [Step 4/6] Installing Python dependencies...
cd /d "%PROJECT_DIR%"
pip install -e .
pip install numpy pandas tqdm networkx python-dotenv tiktoken matplotlib seaborn
echo   √ Python dependencies installed

REM -----------------------------------------------------------
REM 5. Smoke test
REM -----------------------------------------------------------
echo.
echo [Step 5/6] Running smoke tests...
echo   Testing LLM generation...
ollama run llama3.1:7b "Reply with only the word OK" 2>nul
echo   (Check above for LLM response)

REM -----------------------------------------------------------
REM 6. Create directories
REM -----------------------------------------------------------
echo.
echo [Step 6/6] Creating project directories...
if not exist "%PROJECT_DIR%\data\splits" mkdir "%PROJECT_DIR%\data\splits"
if not exist "%PROJECT_DIR%\graphs\graph_full" mkdir "%PROJECT_DIR%\graphs\graph_full"
if not exist "%PROJECT_DIR%\graphs\graph_inc" mkdir "%PROJECT_DIR%\graphs\graph_inc"
if not exist "%PROJECT_DIR%\results\figures" mkdir "%PROJECT_DIR%\results\figures"
echo   √ Directories created

REM -----------------------------------------------------------
REM Final Status
REM -----------------------------------------------------------
echo.
echo ==============================================
echo   SETUP COMPLETE
echo ==============================================
echo.
echo   Next step: python experiments\run_all.py
echo.
pause
