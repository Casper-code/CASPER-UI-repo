import pyautogui
import time
from predict import load_model, predict
from tesseract import ocr
from utils import send_report_email
try:
    from PIL import Image
except ImportError:
    import Image

output_dir = './models/'
model, tokenizer = load_model(output_dir)
#e1 = cv2.getTickCount()
img = pyautogui.screenshot()
lines = ocr(img)
#e2 = cv2.getTickCount()
#time = (e2 - e1) / cv2.getTickFrequency()
#print(time, sum(len(i) for i in lines))
report = []
for line in lines:
    result = predict(model, tokenizer, line)
    if result:
        report.append(line)
send_report_email(report)




