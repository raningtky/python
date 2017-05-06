#-*- coding: UTF-8 -*-
from math import sqrt
def isprime(num):
	if num == 1:
		return False
	for i in range(2,int(sqrt(num))+1):
		if num % i == 0:
			return False
	return True
print filter(isprime, [1, 2, 3, 5, 8, 10])
