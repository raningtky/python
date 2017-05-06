import urllib
filename = urllib.urlretrieve('http://www.baidu.com',filename='/Users/hanxintian/Desktop/baidu.html')
print(type(filename))
print(filename[1])
print(filename[0])
urllib.urlcleanup()
params=urllib.urlencode({'spam':1,'eggs':2,'bacon':0})
f=urllib.urlopen("http://python.org/query?%s" % params)
>>> print f.read()
