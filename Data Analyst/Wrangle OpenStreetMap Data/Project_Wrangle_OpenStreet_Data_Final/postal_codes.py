'''
postal_codes.py
        - Contains functions to validate, audit, process and update
          address:postcode tag
Created on : 1/25/2018
****************************************************************************
'''
from modules import *

'''
Reference : http://zipcode.org/city/CA/SANJOSE
City of San Jose, CA Covers 57 ZIP Codes starting with 951
however since data set was manually exported it might have included near by cities like Cupertino - 95014, 
hence considering those valid
'''
def is_valid_postalCode(postalCode):
    sub_postalCode = postalCode[0:2]
    if not sub_postalCode.isdigit() or sub_postalCode != "95":
        return False
    else:
        return True

'''
Returns true if tag is post code
'''    
def has_postalCode(elem):
    return (elem.attrib['k'] == "addr:postcode")

'''
Iterate the osm xml file for auditing postal code using above functions for
node and way tags
'''
def audit_postalCodes(osmfile):
    osm_file = open(osmfile, "r")
    invalid_postalCodes = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if has_postalCode(tag) and not is_valid_postalCode(tag.attrib['v']):
                    invalid_postalCodes[tag.attrib['v'][0:2]].add(tag.attrib['v'])
	elem.clear()
    return invalid_postalCodes

if __name__ == '__main__':
	cal_postalCode = audit_postalCodes(sanjose_data)
	pprint.pprint (dict(cal_postalCode))
