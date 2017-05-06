# -*- coding: utf-8 -*-
# CopyRight by heibanke

import urllib
from bs4 import BeautifulSoup
import re


url='http://www.heibanke.com/lesson/crawler_ex00/'
number=['']
loops = 0

while True:
    content = urllib.urlopen(url+number[0])

    bs_obj = BeautifulSoup(content,"html.parser")
    tag_number = bs_obj.find("h3")

    number= re.findall(r'\d+',tag_number.get_text())
    
    if not number or loops>100:
        break
    else:
        print number[0]

    loops+=1


print bs_obj.text