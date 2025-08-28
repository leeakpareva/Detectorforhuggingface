# 🔧 NAVADA Troubleshooting Guide

## 🌐 Public Link Issues

### Problem: "Could not create share link. Missing file: frpc_windows_amd64_v0.3"

**Cause**: Your antivirus software is blocking Gradio's tunneling client download.

**Solutions**:

#### Option 1: Add Antivirus Exclusion (Recommended)
1. Open your antivirus software (Windows Defender, McAfee, Norton, etc.)
2. Add these folders to exclusions:
   - `C:\Users\[username]\.cache\huggingface\`
   - `C:\Users\[username]\.cache\gradio\`
3. Restart your application

#### Option 2: Manual Download
1. Create directory: `C:\Users\[username]\.cache\huggingface\gradio\frpc\`
2. Download: https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_windows_amd64
3. Rename to: `frpc_windows_amd64_v0.3` (no extension)
4. Make executable: `chmod +x frpc_windows_amd64_v0.3`

#### Option 3: Use ngrok (Alternative)
1. Install ngrok: https://ngrok.com/
2. Sign up and get auth token
3. Run: `ngrok config add-authtoken YOUR_TOKEN`
4. Use our helper script: `setup_public_link.bat`
5. Or manually run: `ngrok http 7860`

#### Option 4: Use Cloudflare Tunnel
```bash
# Install cloudflared
winget install --id Cloudflare.cloudflared

# Create tunnel
cloudflared tunnel --url localhost:7860
```

---

## 🚫 Common Antivirus Blocks

### Windows Defender
1. Open **Windows Security** → **Virus & threat protection**
2. Click **Manage settings** under "Virus & threat protection settings"
3. Scroll to **Exclusions** → **Add or remove exclusions**
4. Add folder: `C:\Users\[username]\.cache\`

### McAfee
1. Open McAfee → **Virus and Spyware Protection**
2. Click **Real-Time Scanning** → **Turn Off**
3. Or add exclusion in **Quarantined and Trusted Items**

### Norton
1. Open Norton → **Settings** → **Antivirus**
2. Go to **Scans and Risks** → **Exclusions/Low Risks**
3. Add folder exclusion

### Avast/AVG
1. Open Avast → **Protection** → **Core Shields**
2. **File Shield** → **Configure** → **Exclusions**
3. Add folder path

---

## 🐍 Python Environment Issues

### Problem: ModuleNotFoundError
```bash
# Create virtual environment
python -m venv navada_env

# Activate (Windows)
navada_env\Scripts\activate

# Activate (Mac/Linux)
source navada_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Problem: CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔑 OpenAI API Issues

### Problem: Invalid API Key
1. Check `.env` file exists and has correct format:
   ```
   OPENAI_API_KEY=sk-...your-key-here...
   ```
2. Verify key at: https://platform.openai.com/api-keys
3. Check billing: https://platform.openai.com/account/billing

### Problem: Rate Limits
- Upgrade OpenAI plan
- Add retry logic with exponential backoff
- Use `gpt-3.5-turbo` instead of `gpt-4` for lower costs

---

## 📱 Port and Network Issues

### Problem: Port 7860 in use
```bash
# Find process using port
netstat -ano | findstr :7860

# Kill process (replace PID)
taskkill /PID <PID> /F

# Use different port
python app.py --port 7861
```

### Problem: Firewall blocking
1. Windows: **Windows Defender Firewall** → **Allow an app**
2. Add Python to allowed apps
3. Or disable firewall temporarily for testing

---

## 🎥 Webcam/Camera Issues

### Problem: Camera not accessible
1. Check permissions: **Settings** → **Privacy** → **Camera**
2. Close other apps using camera
3. Restart camera service:
   ```bash
   net stop "Windows Camera Frame Server"
   net start "Windows Camera Frame Server"
   ```

---

## 🔍 Model Loading Issues

### Problem: YOLOv8 model download fails
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Or use different model
pip install ultralytics
yolo download model=yolov8n.pt
```

### Problem: OpenCV Haar Cascades missing
```bash
# Verify OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Reinstall if needed
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

---

## 🌍 Browser Issues

### Problem: App not loading
1. Clear browser cache and cookies
2. Try different browser (Chrome, Firefox, Edge)
3. Disable browser extensions
4. Check if localhost:7860 is accessible

### Problem: JavaScript errors
1. Enable JavaScript in browser
2. Disable ad blockers
3. Check browser console for errors (F12)

---

## 📊 Performance Issues

### Problem: Slow processing
1. **GPU Acceleration**: Install CUDA-enabled PyTorch
2. **Model Size**: Use YOLOv8n (nano) for speed
3. **Image Size**: Resize large images before processing
4. **Batch Processing**: Process multiple images together

### Problem: High memory usage
1. **Reduce batch size**
2. **Use smaller model variants**
3. **Clear cache periodically**
4. **Monitor with Task Manager**

---

## 🚨 Emergency Fixes

### Quick Reset
```bash
# Stop all Python processes
taskkill /F /IM python.exe

# Clear cache
rmdir /S "%USERPROFILE%\.cache\huggingface"
rmdir /S "%USERPROFILE%\.cache\gradio"

# Restart app
python app.py
```

### Factory Reset
```bash
# Remove virtual environment
rmdir /S navada_env

# Fresh installation
python -m venv navada_env
navada_env\Scripts\activate
pip install -r requirements.txt
python app.py
```

---

## 📞 Get Help

If none of these solutions work:

1. **GitHub Issues**: https://github.com/leeakpareva/Detectorforhuggingface/issues
2. **Gradio Forum**: https://discuss.huggingface.co/
3. **Stack Overflow**: Tag questions with `gradio`, `yolov8`, `opencv`

### Include in Bug Reports
- Operating system and version
- Python version (`python --version`)
- Complete error message
- Steps to reproduce
- Screenshots if applicable

---

## 🔧 Development Mode

For developers who want to modify the app:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Format code
black *.py backend/*.py

# Run tests
pytest tests/

# Enable debug mode
export GRADIO_DEBUG=1  # Linux/Mac
set GRADIO_DEBUG=1     # Windows
```

---

*Last Updated: December 2024*
*For more help, visit our GitHub repository or contact support*