# import pyautogui
# import threading
# from predict import load_model, predict
# from tesseract import textDetection
# from utils import send_report_email, check_config_file
# try:
#     from PIL import Image
# except ImportError:
#     import Image
#
#
# def text_detection_ocr():
#     check_config_file()
#     while True:
#         img = pyautogui.screenshot()
#         #img.save(r"screenshot.png")
#         lines = textDetection(img)
#         report = []
#         for line in lines:
#             result = predict(model, tokenizer, line)
#             if result:
#                 report.append(line)
#         if report:
#             print('Sending report...')
#             send_report_email(report)
#
#
# output_dir = './models/'
# model, tokenizer = load_model(output_dir)
# text_detection_ocr = threading.Thread(name='text_detection_ocr', target=text_detection_ocr)
# text_detection_ocr.start()
#
#
