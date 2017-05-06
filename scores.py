#!/usr/bin/env python3
#coding: utf-8
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import pandas as pd
def sendemail(number,score):
    sender = 'hxt1108@163.com'
    #receiver = str(number) + '@pku.edu.cn'
    receiver = 'hxt1108@163.com'
    subject = '期中考试成绩'
    smtpserver = 'smtp.163.com'
    username = 'hxt1108@163.com'
    password = 'han13868134817xt'
    text = '如果您的学号是' + str(number) + ','
    text += '您的成绩是' + str(score) +'\n'
    msg = MIMEText(text, 'plain', 'utf-8')
    msg['Subject'] = Header(subject,'utf-8')
    smtp = smtplib.SMTP()
    smtp.connect('pku.edu.cn')
    smtp.login(username,password)
    smtp.sendemail(sender, receiver, msg.as_string())
    smtp.quit()

X = pd.read_csv('~/Desktop/scores.csv')
sendemail(1300017608,75)

