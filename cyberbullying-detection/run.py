import pyautogui
import threading
from predict import load_model, predict
from tesseract import ocr
from utils import send_report_email
try:
    from PIL import Image
except ImportError:
    import Image


def text_detection_ocr():
    while True:
        img = pyautogui.screenshot()
        #img.save(r"screenshot.png")
        lines = ocr(img)
        report = []
        for line in lines:
            result = predict(model, tokenizer, line)
            if result:
                report.append(line)
        if report:
            print('Sending report...')
            send_report_email(report)


output_dir = './models/'
model, tokenizer = load_model(output_dir)
text_detection_ocr = threading.Thread(name='text_detection_ocr', target=text_detection_ocr)
text_detection_ocr.start()


