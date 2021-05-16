#!/usr/bin/python
'''
street_types.py
	- Contains functions to validate, audit, process and update
	  address:street and address:full tags
	- get count if street types to identify issues in the field
Created on : 1/25/2018
****************************************************************************
'''

from modules import *

if __name__ == '__main__':
	t0 = time()

def getContext(filename):
	return ET.iterparse(filename)
	
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

#Reference : https://en.wikipedia.org/wiki/Street_or_road_name
EXPECTED_STREET_TYPE = ["Avenue", "Boulevard", "Commons", "Court", "Drive", "Lane", "Parkway", "Place", "Road", "Square", "Street", "Trail", "Terrace",
            "Expressway" , "Circle", "Way", "Loop" ] 

'''
Check if tag attrib 'k' is 'addr:street' or 'addr:full'
'''
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street" or elem.attrib['k'] == "addr:full")

# Get count of street types to identify all issues with street names
'''
Get count of street types as a dict only incase of node and way tags
'''
def get_street_types_count(filename):
    streetTypesCount = defaultdict(int)
    for _, elem in ET.iterparse(filename, events=('start',)):
        if elem.tag == 'node' or elem.tag == 'way':
            for tag in elem.iter('tag'):
                if is_street_name(tag):
                    m = street_type_re.search(tag.attrib['v'])
                    if m:
                        streetTypesCount[m.group()] += 1
    return streetTypesCount


if __name__ == '__main__':
	streetTypesCount = get_street_types_count(sanjose_data)
	print '--street types--'
	pprint.pprint(dict(streetTypesCount))


'''
Searches the input string for the regex. 
If there is a match i.e last string in tag and it is not within the "expected" list, 
add the match as a key and add the string to the set.
'''
def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group(0)
        if street_type not in EXPECTED_STREET_TYPE:
            street_types[street_type].add(street_name)
            
'''
This function will return the list that matches is street_name and audit_street_type functions. 
After that, we would do a pretty print the output of the audit.
''' 
def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for _, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
        elem.clear() #Memory cleanup
    return street_types

if __name__ == '__main__':
	cal_street_types = audit(sanjose_data)
	print (dict(cal_street_types))


'''
Create mapping dictionary based on above results from cal_street_types
'''
MAP_STREET_TYPE = \
{          'Ave'  : 'Avenue',
           'Blvd' : 'Boulevard',
           'Blvd.' : 'Boulevard',
           'Dr'   : 'Drive',
           'Ln'   : 'Lane',
           'Pkwy' : 'Parkway',
           'Rd'   : 'Road',
           'Rd.'   : 'Road',
           'St'   : 'Street',
           'street' :"Street",
           'Ct'     : "Court",
           'Cir'    : "Circle",
           'Cr'     : "Court",
           'ave'   : 'Avenue',
           'Hwg'  : 'Highway',
           'Hwy'  : 'Highway',
           'Sq'    : "Square",
           'Ter'   : "Terrace",
	   'Expy'  : "Expressway",
	   'Ste'   : "Suite" 
}

'''
Update street name using mapping dict.
'''
def update_street_name(name, mapping, regex):
    m = regex.search(name)
    if m and m.group() in mapping.keys():
        map_name = mapping[m.group()]
        name = street_type_re.sub(map_name, name).strip()
    return name

if __name__ == '__main__':
	for street_type, ways in cal_street_types.iteritems():
    		for name in ways:
        		better_name = update_street_name(name, MAP_STREET_TYPE, street_type_re)
        		print name, "=>", better_name
	print os.path.basename(__file__) ,"script Ended.Duration: ", round(time()-t0,3), "s"
			
