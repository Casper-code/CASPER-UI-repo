import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import configparser
import ctypes


def check_config_file():
    config = configparser.RawConfigParser()
    config.read('email.config')
    sender_email = config.get('email_info', 'sender_email')
    receiver_email = config.get('email_info', 'receiver_email')
    password = config.get('email_info', 'password')
    all_ok = True
    if sender_email is None or sender_email == "":
        ctypes.windll.user32.MessageBoxW(None, u"You must enter sender_email in email.config", u"Error", 0)
        all_ok = False
    if receiver_email is None or receiver_email == "":
        ctypes.windll.user32.MessageBoxW(None, u"You must enter receiver_email in email.config", u"Error", 0)
        all_ok = False
    if password is None or password == "":
        ctypes.windll.user32.MessageBoxW(None, u"You must enter password in email.config", u"Error", 0)
        all_ok = False
    if not all_ok:
        raise ValueError('Error in email.config file')


def send_report_email(lines):
    config = configparser.RawConfigParser()
    config.read('email.config')
    sender_email = config.get('email_info', 'sender_email')
    receiver_email = config.get('email_info', 'receiver_email')
    password = config.get('email_info', 'password')

    message = MIMEMultipart("alternative")
    message["Subject"] = "Automatic cyberbullying detection report"
    message["From"] = "CASPER Agent"
    message["To"] = receiver_email

    # Create the plain-text and HTML version of your message
    html = "<html><body>Dear parent,<br/>" \
           "We have detected activity in the context of cyberbullying on the social media channels which involves " \
           "your child: "

    for line in lines:
        html += "<p>" + line + "</p>"
    html += "CASPER Agent</body></html>"
    part = MIMEText(html, "html")
    message.attach(part)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_email, message.as_string()
        )
