import os
import subprocess
import sys

# Check if file exists
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
print(f"File exists: {os.path.exists(tesseract_path)}")
print(f"File path: {tesseract_path}")

# Try running it directly
try:
    result = subprocess.run([tesseract_path, '--version'], capture_output=True, text=True)
    print("✓ Direct subprocess works:")
    print(result.stdout)
except Exception as e:
    print(f"✗ Direct subprocess failed: {e}")

# Try pytesseract
try:
    import pytesseract
    pytesseract.pytesseract.pytesseract_cmd = tesseract_path
    version = pytesseract.get_tesseract_version()
    print(f"✓ Pytesseract works: {version}")
except Exception as e:
    print(f"✗ Pytesseract failed: {e}")