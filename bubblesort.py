def bubbleSort(nums):
	for j in range(len(nums)):
		for i in range(len(nums))
			if nums[i] > nums[i+1]:
				nums[i],nums[i+1] = nums[i+1],nums[i]
#TEST
if __name__ == '__main__':
	numbers=[[9,23,12,32,12],['2','3','3','6'],['b','w','u']]
	for num in numbers:
		bubbleSort(num)
		print num