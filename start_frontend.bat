@echo off
echo Starting Frontend Server...
echo.
echo Frontend will be available at: http://localhost:8000
echo Backend API is running at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
cd Code\frontend
python -m http.server 8000
