import pyautogui

from textDetection.predict import load_model, predict
from textDetection.tesseract import ocr
from textDetection.utils import send_report_email, check_config_file
from workers.worker import WorkerThread
import tqdm
import torch
from os import path

try:
    from PIL import Image
except ImportError:
    import Image

output_dir = './textDetection/models/'
model, tokenizer = load_model(output_dir)


class TextDetection(WorkerThread):

    def __init__(self, *args, **kwargs):
        super(TextDetection, self).__init__(*args, **kwargs)

    def process_iteration(self):
        check_config_file()
        while True:
            img = pyautogui.screenshot()
            # img.save(r"screenshot.png")
            lines = ocr(img)
            report = []
            for line in lines:
                result = predict(model, tokenizer, line)
                if result:
                    report.append(line)
            if report:
                print('Sending report...')
                send_report_email(report)
