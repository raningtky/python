#! /usr/bin/env python  
# This Python file uses the following encoding: utf-8
import smtplib
from email.mime.text import MIMEText
import pandas as pd
mail_host="smtp.163.com"            #使用的邮箱的smtp服务器地址
mail_user="hxt1108"                           #用户名
mail_pass="********"                             #密码
mail_postfix="163.com"                     #邮箱的后缀
def send_mail(number,score):
    me="hello"+"<"+mail_user+"@"+mail_postfix+">"
    text = '如果您的学号是' + str(number) + ','
    text += '您的成绩是' + str(score) +'\n'
    to_list = str(number) + '@pku.edu.cn'
    print to_list
    msg = MIMEText(text, 'plain', 'utf-8')
    msg['Subject'] = 'mid-term score'
    msg['From'] = me
    try:
        server = smtplib.SMTP()
        server.connect(mail_host)                            #连接服务器
        server.login(mail_user,mail_pass)               #登录操作
        server.sendmail(me, to_list, msg.as_string())
        server.close()
        return True
    except Exception, e:
        print str(e)
        return False
X = pd.read_csv('~/Desktop/scores.csv')
for i in range(0,len(X)):                             
    number = str(int(X.iloc[i,0]))
    score = str(X.iloc[i,1])
    if score != 'nan':
        print str(int(number)),str(score)
        send_mail(number,str(score))
