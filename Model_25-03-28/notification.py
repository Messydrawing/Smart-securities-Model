import smtplib
from email.mime.text import MIMEText
from email.header import Header

def send_email(to_addr, subject, body):
    from_addr = "???@xx.com"
    password = "???"  # 邮箱授权码
    smtp_server = "smtp.xx.com"

    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = from_addr
    msg['To'] = to_addr

    try:
        # 1) 使用 SMTP_SSL 并指定 465 端口
        server = smtplib.SMTP_SSL(smtp_server, 465)
        # 2) 不要再调用 starttls()，否则会握手失败
        server.login(from_addr, password)
        server.sendmail(from_addr, [to_addr], msg.as_string())
        server.quit()
        print(f"✅ 已向 {to_addr} 发送邮件：{subject}")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
