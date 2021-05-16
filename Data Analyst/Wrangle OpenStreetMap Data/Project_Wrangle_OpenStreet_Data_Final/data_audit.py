#!/usr/bin/python
'''
data_audit.py
 - Investigate openstreetmap data set
 - Parses the original osm xml data to
	- Data overview and get count for top level elements
	- Explore any problematic characters in attribute named 'k' within tag
	- Explore different address types within k attribute
	- Validate whether the inner level tags can be used as valid MongoDB keys using random sample. 
Created on : 1/25/2018
****************************************************************************
'''

from modules import *

t0 = time()

def getContext(filename):
	return ET.iterparse(filename)
	

#Step 1 : Data Overview
#Print the data set name along with size
pprint.pprint("Original OSM XML File : " + sanjose_data + ", Size : " + str(os.path.getsize(sanjose_data) >> 20) + " MB") 

root = ET.parse(sanjose_data).getroot()
#Map coordinates
print "Map Coordinates:"
for elem in root.findall('bounds'):
        print elem.tag, elem.attrib
	elem.clear()

'''
Parse through the file with ElementTree and 
count the number of top level element types to 
understand overall structure.
Returns dictionary of (key)top level elements and (value) their count.
'''
def get_tag_count(filename):
        tags = defaultdict(int)
	context = getContext(filename)
        for _, elem in context:
            if elem.tag in tags: 
                tags[elem.tag] += 1
            else:
                tags[elem.tag] = 1
	elem.clear()

	del context
        return tags

#Save and prints the unique top level element types and their count    
sanjose_tags = get_tag_count(sanjose_data)
pprint.pprint("Unique Top Level Elements:")
pprint.pprint(dict(sanjose_tags))

#Step 2 : Explore any problematic characters in attribute named 'k' within tag
'''
A tag consists of two items, a key and a value. 
Tags describe specific features of map elements (nodes, ways, or relations) or changesets. 
Both items are free format text fields, but often represent numeric or other structured items
In order to extract information from this tag, we need to ensure it does not have characters issues. 

Sample OSM xml:
# Search for element types named tag within osm xml and gets the attributes named 'k' within tag e.g.
# <node id="266587529" lat="37.3625767" lon="-122.0251570" version="4" timestamp="2015-03-30T03:17:30Z" changeset="29840833" uid="2793982" user="Dhruv Matani">
#     <tag k="addr:city" v="Sunnyvale"/>
#     <tag k="addr:housenumber" v="725"/>
#     <tag k="addr:postcode" v="94086"/>
#     <tag k="addr:state" v="California"/>
#     <tag k="addr:street" v="South Fair Oaks Avenue"/>
#     <tag k="amenity" v="restaurant"/>
#     <tag k="cuisine" v="indian"/>
#     <tag k="name" v="Arka"/>
#     <tag k="opening_hours" v="10am - 2:30pm and 5:00pm - 10:00pm"/>
#     <tag k="takeaway" v="yes"/>
#  </node>
'''
lower = re.compile(r'^([A-Z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
 
'''
Returns dictionary as {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0} with counts as below:
"lower" - for tags that contain only lowercase letters and are valid, 
"lower_colon" - for otherwise valid tags with a colon in their names, 
"problemchars" - for tags with problematic characters, and
"other" - for tags which do not fall into above category.
'''
def categorize_key_type(element, mydict, attrib):
    if element.tag == "tag":
        for tag in element.iter('tag'):
            attribVal = tag.get(attrib)
            if lower.search(attribVal):
                mydict['lower'] += 1
            elif lower_colon.search(attribVal):
                mydict['lower_colon'] += 1
            elif problemchars.search(attribVal):
                mydict['problemchars'] += 1
            else:
                mydict['other'] += 1
    return mydict

'''
Parse through every element the dataset to identify anamolies in element tag
'''
def process_xml(filename, attrib):
    mydict = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    context = getContext(filename)
    for event, elem in context:
        mydict = categorize_key_type(elem, mydict, attrib)
	elem.clear()

    del context
    return mydict

sanjose_keyDict = process_xml(sanjose_data,'k')
print "Character types in k tag:"
pprint.pprint(dict(sanjose_keyDict))

# Step 3 - Explore different address types within k attribute
'''
Iterate xml for node and way tags
for tag k within node and way, check if it startswith addr
Returns a dictionary for those tag along with their occurance count
'''
def find_addr_types(filename):
    addr_types = {}
    context = getContext(filename)
    for _, elem in context:
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                key_val = tag.get("k")
                if key_val is not None and key_val.lower()[:4]  == "addr":
                    if key_val in addr_types:
                        addr_types[key_val] +=1
                    else:
                        addr_types[key_val] =1
    del context
    return addr_types

addr_types = find_addr_types(sanjose_data)
print "Different Address types within k attribute"
pprint.pprint(dict(addr_types))

# Step 4 - Validate whether the inner level tags can be used as valid MongoDB keys using random sample.
# dumping key/value pairs for tag attribute k,v  with a selection of random sample  to identify potential probles in dataset
'''
Get tag data as a dictionary with k, v pairs for node and way tags
'''
def get_tags(filename):
    tagData = defaultdict(set)
    for _, elem in ET.iterparse(filename, events=('start',)):
        if elem.tag == 'node' or elem.tag == 'way':
            for tag in elem.iter('tag'):
                tagData[tag.attrib['k']].add(tag.attrib['v'])
        elem.clear() # Memory cleanup
    return tagData

'''
Dump a random sample of tags to file.
Limit will determine the max number of values for each tag.
'''
def dump_tags(filename, limit=10):
    tagData = get_tags(filename)
    # converting tagData to list (so that it can be dumped into json)
    for key in tagData.keys():
        tagData[key] = list(tagData[key])
        
    if limit >= 0:
        # randomly select values based on limit
        tagData_dump = defaultdict(list)
        for key in tagData.keys():
            if len(tagData[key]) < limit:
                tagData_dump[key] = random.sample(tagData[key], len(tagData[key]))
            else:
                tagData_dump[key] = random.sample(tagData[key], limit)
    elif limit < 0:
        # dump all data
        tagData_dump = tagData
    
    # dump to json file
    json.dump(tagData_dump, open(sample_json_data, 'w'), indent=4, sort_keys=True)

dump_tags(sanjose_data,50)

#Print the data set name along with size
pprint.pprint("Sample JSON File : " + sample_json_data + ", Size : " + str(os.path.getsize(sample_json_data)) + " bytes")

print os.path.basename(__file__) ,"script Ended.Duration: ", round(time()-t0,3), "s"

