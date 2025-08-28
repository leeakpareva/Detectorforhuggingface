@echo off
echo 🚀 NAVADA Public Link Setup
echo.

REM Check if ngrok is installed
where ngrok >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ ngrok not found. Please install ngrok first:
    echo    1. Go to https://ngrok.com/
    echo    2. Sign up for a free account
    echo    3. Download and install ngrok
    echo    4. Run: ngrok config add-authtoken YOUR_TOKEN
    echo.
    pause
    exit /b 1
)

echo ✅ ngrok found!
echo 🌐 Creating public tunnel for NAVADA...
echo.
echo 📋 Instructions:
echo    1. Keep this window open
echo    2. Start NAVADA in another terminal: python app.py
echo    3. Use the ngrok URL below to share your app
echo.

REM Start ngrok tunnel
ngrok http 7860