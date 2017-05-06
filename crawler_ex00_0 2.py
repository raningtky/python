# -*- coding: utf-8 -*-
# CopyRight by heibanke

import urllib
from bs4 import BeautifulSoup

url='http://www.heibanke.com/lesson/crawler_ex00/'
content = urllib.urlopen(url)

bs_obj = BeautifulSoup(content,"html.parser")


numberstr = bs_obj.find("h3")

print numberstr.get_text()

import re

print re.findall(r'\d+',numberstr.get_text())