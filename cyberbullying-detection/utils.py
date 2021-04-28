import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_report_email(lines):
    sender_email = "casper2.project@gmail.com"
    receiver_email = ""
    password = ""

    message = MIMEMultipart("alternative")
    message["Subject"] = "Automatic cyberbullying detection report"
    message["From"] = "CASPER Agent"
    message["To"] = ""

    # Create the plain-text and HTML version of your message
    html = "<html><body>Dear parent,<br/>" \
           "We have detected activity in the context of cyberbullying on the social media channels which involves your child:"

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

