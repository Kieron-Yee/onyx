"""
此模块提供了电子邮件相关的工具函数，主要用于:
1. 发送系统邮件
2. 发送用户邀请邮件
3. 发送密码重置邮件
4. 发送邮箱验证邮件
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from textwrap import dedent

from onyx.configs.app_configs import EMAIL_CONFIGURED
from onyx.configs.app_configs import EMAIL_FROM
from onyx.configs.app_configs import SMTP_PASS
from onyx.configs.app_configs import SMTP_PORT
from onyx.configs.app_configs import SMTP_SERVER
from onyx.configs.app_configs import SMTP_USER
from onyx.configs.app_configs import WEB_DOMAIN
from onyx.db.models import User


def send_email(
    user_email: str,
    subject: str,
    body: str,
    mail_from: str = EMAIL_FROM,
) -> None:
    """
    发送电子邮件的通用函数
    
    参数:
        user_email: 收件人邮箱地址
        subject: 邮件主题
        body: 邮件正文内容
        mail_from: 发件人邮箱地址，默认使用系统配置的地址
        
    返回:
        None
        
    异常:
        ValueError: 当邮件配置不完整时抛出
        Exception: 当发送邮件失败时抛出
    """
    if not EMAIL_CONFIGURED:
        raise ValueError("Email is not configured.")

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["To"] = user_email
    if mail_from:
        msg["From"] = mail_from

    msg.attach(MIMEText(body))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
    except Exception as e:
        raise e


def send_user_email_invite(user_email: str, current_user: User) -> None:
    """
    发送用户邀请邮件
    
    向新用户发送加入组织的邀请邮件，邮件中包含注册链接
    
    参数:
        user_email: 被邀请用户的邮箱地址
        current_user: 发起邀请的当前用户对象
        
    返回:
        None
    """
    subject = "Invitation to Join Onyx Organization"
    body = dedent(
        f"""\

        Hello,

        You have been invited to join an organization on Onyx.

        To join the organization, please visit the following link:

        {WEB_DOMAIN}/auth/signup?email={user_email}

        You'll be asked to set a password or login with Google to complete your registration.

        Best regards,
        The Onyx Team
    """
    )

    send_email(user_email, subject, body, current_user.email)


def send_forgot_password_email(
    user_email: str,
    token: str,
    mail_from: str = EMAIL_FROM,
) -> None:
    """
    发送忘记密码邮件
    
    向用户发送密码重置链接，链接中包含重置token
    
    参数:
        user_email: 用户邮箱地址
        token: 密码重置token
        mail_from: 发件人邮箱地址，默认使用系统配置的地址
        
    返回:
        None
    """
    subject = "Onyx Forgot Password"
    link = f"{WEB_DOMAIN}/auth/reset-password?token={token}"
    body = f"Click the following link to reset your password: {link}"
    send_email(user_email, subject, body, mail_from)


def send_user_verification_email(
    user_email: str,
    token: str,
    mail_from: str = EMAIL_FROM,
) -> None:
    """
    发送邮箱验证邮件
    
    向用户发送邮箱验证链接，用于验证用户邮箱的有效性
    
    参数:
        user_email: 待验证的用户邮箱地址
        token: 验证token
        mail_from: 发件人邮箱地址，默认使用系统配置的地址
        
    返回:
        None
    """
    subject = "Onyx Email Verification"
    link = f"{WEB_DOMAIN}/auth/verify-email?token={token}"
    body = f"Click the following link to verify your email address: {link}"
    send_email(user_email, subject, body, mail_from)
