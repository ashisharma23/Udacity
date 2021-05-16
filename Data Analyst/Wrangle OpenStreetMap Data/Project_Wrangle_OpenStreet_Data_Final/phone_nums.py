import re
'''
phone_nums.py
        - Contains functions to validate, audit, process and update
          phone number value , phone tag
Created on : 1/25/2018
****************************************************************************
'''

'''
Formats phone number to standard formats: 1XXXXXXXXXX
'''
def update_phone_number(number):
	'''
	retain digits only (removing + also)
	Removing left-to-right and right-to-left marker, as seen in one of the data fields.
	replace(u"\u200e", u"").replace(u"\u200f", u"") -does not work since its now just a mere string.
	'''
	number = re.sub('[^\d]', '', number.replace('u200e', u"").replace('u200f', u""))
    	# add +1 to the beginning
    	if len(number) == 10:
        	number = '+1' + number
    	elif len(number) == 11 and number.startswith('1'):
        	number = '+' + number
    	else:
        	number = None
    	return number

def test():
	numbers = ['4089882555', '+1 408 971 3977', '(408) 354-7365', '+1 408 988 1883','+1-408-971-1160', \
	'+1.408.736.6688','+1-408-988-5407','408.266.6342','+1 408-500-3000 \u200e']
	print "--Update phone number examples--"
	for num in numbers:
    		print '  ', num, '=>', update_phone_number(num)

if __name__ == '__main__':
	test()
