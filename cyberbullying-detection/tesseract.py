import pytesseract
# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def ocr(img):
    text = pytesseract.image_to_string(img, lang='eng')
    print(text)
    text = text.split('\n\n')
    lines = []
    for line in text:
        if len(line) >= 4:
            lines.append(line)
    return lines
