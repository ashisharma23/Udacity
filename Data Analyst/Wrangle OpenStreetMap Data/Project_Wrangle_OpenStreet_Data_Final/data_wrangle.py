'''
data_wrangle.py
        - Contains functions to process the osm xml and reshape it into json format
        - 
Created on : 1/25/2018
****************************************************************************
'''

from modules import *
from street_types import update_street_name, MAP_STREET_TYPE,street_type_re
from phone_nums import update_phone_number
print os.path.basename(__file__) ,"script Started.."

t0 = time()
lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
address_regex = re.compile(r'^addr\:')
tiger_regex = re.compile(r'^tiger\:')
street_regex = re.compile(r'^street|^full')

#Created dictionary keys
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

#Cuisine Mapping for categorization
CUISINE_MAPPING = {
"hotdog" : "american",
"hot_dog" : "american",
"mediterranean" : "Mediterranean",
"sandwiches"    : "sandwich",
"sushi" : "japanese",
"_pizza" : "italian",
"Traditional American" : "american",
"burger" : "american",
"ice_cream" : "dessert",
"frozen_yogurt" : "dessert",
"_burger" : "american",
"_steak" : "american",
"noodles" : "chinese",
"pearl_tea" : "Coffee & Tea",
"boba_tea" : "Coffee & Tea",
"coffee_shop" : "Coffee & Tea",
"Coffee" : "Coffee & Tea",
"bubble_tea" : "Coffee & Tea",
"tea" : "Coffee & Tea",
"afgan" : "middle_eastern",
"boba" : "asian",
"vietnamnese" : "asian",
"vietnamese_Restaurant" : "asian",
"vietnamese" : "asian",
"greek" : "Mediterranean",
"afghan" : "middle eastern",
"middle" : "middle_eastern",
"arab" : "middle_eastern",
"burmese" : "asian",
"taco" : "mexican",
"donut" : "american",
"donuts" : "american",
"thai_fusion" : "thai",
"arab" : "middle_eastern",
"falafel" : "mediterranean",
"sandwich" : "british",
"english" : "british",
"taiwanese" : "asian",
"korean" : "asian",
"pizza" : "italian"
}

'''
Function checks if any item of the list is present in CUISINE_MAPPING dictionary
If yes, it replaces the same with mapped value.
'''
def apply_cuisineMapping(cus_list):
	new_list = []
	for x in cus_list:
        	if x in CUISINE_MAPPING:
                	new_list.append(CUISINE_MAPPING[x])
		else:
                  	new_list.append(x)
	return new_list

'''
Function transform node and way elements in original osm xml into dictionary(node), 
so later can be dumped in json file.
	- Process only 'node' and 'way' tags, else returns None
	- Creates a empty Dict - 'node'
	- parses element tag to create below items within node
		- "created" dict
		- "pos" dict with latitude and longitude information
		- "node_ref" with 2nd level tags
		- ignore not tag elements and without 'k' and 'v'
		- transform phone number in standard format
		- transform cuisine into list format and standarized values
		- expand address:street and address:full values for abbreviations
		- update postal code
		- transform tiger tags into dictionary format
'''
def shape_element(element):
    node = {}
    if element.tag == "node" or element.tag == "way" :
        node['type'] = element.tag
        # initialize empty address
        address = {}
        tiger_dict = {}
        # parsing through attributes
        for a in element.attrib:
            if a in CREATED:
                if 'created' not in node:
                    node['created'] = {}
                node['created'][a] = element.get(a)
            elif a in ['lat', 'lon']:
                continue
            else:
                node[a] = element.get(a)
        # populate position
        if 'lat' in element.attrib and 'lon' in element.attrib:
            node['pos'] = [float(element.get('lat')), float(element.get('lon'))]

        # parse second-level tags for nodes
        for e in element:
            # parse second-level tags for ways and populate `node_refs`
            if e.tag == 'nd':
                if 'node_refs' not in node:
                    node['node_refs'] = []
                if 'ref' in e.attrib:
                    node['node_refs'].append(e.get('ref'))

            # throw out not-tag elements and elements without `k` or `v`
            if e.tag != 'tag' or 'k' not in e.attrib or 'v' not in e.attrib:
                continue
            key = e.get('k').strip()
            val = e.get('v').strip()
	    if key == 'type': #this ensures way and node types are not overridden with tag k='type'
		continue
	    if key.find("disused:") != -1: #Ignore if key is tagged as disused.
		continue	
            # skip problematic characters
            if problemchars.search(key):
                continue

            # parse address k-v pairs
            elif address_regex.search(key):
                key = key.replace('addr:', '')
                address[key] = val
            # parse address k-v pairs
            elif tiger_regex.search(key):
                key = key.replace('tiger:', '')
                tiger_dict[key] = val
            #Format phone numbers
            elif key =="phone":
                val = update_phone_number(val)
                node[key] = val
	    elif key == "cuisine":
		if val.find(",") != -1:
			val = val.split(",") # Lot of cusinies are described as , or ; seperated.
		else:
		   	val = val.split(";") # This will also handle case when there is no delimiter
		val = apply_cuisineMapping(val)
		node[key] = val
	    elif key == "name":  #Standarizing name of popular restaurants and cafes
		if val.find("Peet") != -1 and val.find("Coffee") != -1:
			node[key] = "Peet's Coffee"
		elif val.find("Starbucks") != -1:
			node[key] = "Starbucks Coffee"
		elif val.find("Round Table") != -1:
			node[key] = "Round Table Pizza"
		else:
			node[key] = val
            else:
                node[key] = val
        # compile address
        if len(address) > 0:
            node['address'] = {}
            street_full = None
	    isKeyStreet_exists = False
	    isKeyFull_exists = False	
            street_dict = {}
            street_format = ['prefix', 'name', 'type']
            # parse through address objects
            for key in address:
                val = address[key]
                if street_regex.search(key):
		    if key == 'street':
			isKeyStreet_exists = True
		    if key == 'full':
			isKeyFull_exists = True
                    if key == 'street' or key =='full':
                        street_full = val
                        street_full  = update_street_name(street_full, MAP_STREET_TYPE, street_type_re)
			key_text = key + ":"
                        street_dict[key.replace(key_text, '')] = street_full
                #During postal code auditing , we find few postal code starts with 'CA'        
                elif key == 'postcode': 
			if val.startswith('CA'):
                    		val = val.replace('CA ',"") # During manual review, it was found few post code starts with 'CA '
			elif val.startswith('CU'): # One postal code was updated as 'CUPERTINO' - ignoring that value.
				val = ""
                    	node['address'][key] = val
                else:
                    node['address'][key] = val
            # assign street_full or fallback to compile street dict
            if street_full and isKeyStreet_exists:
                node['address']['street'] = street_full
	    elif street_full and isKeyFull_exists:
		node['address']['full'] = street_full
            elif len(street_dict) > 0:
                node['address']['street'] = ' '.join([street_dict[key] for key in street_format])
        if  len(tiger_dict) > 0:      
            node['tiger'] = tiger_dict   
        return node
    else:
        return None

'''
Iterate osm xml file_in by calling shape_element for each element and 
dump the json output into file_out 
'''
def process_map(file_in,file_out, pretty = False):
    data = []
    counter = 0
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el :
                counter = counter + 1
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+'\n')
                else:
                    fo.write(json.dumps(el) + '\n')
    return data

data  = process_map(sanjose_data,clean_json_data, False)

#Print the data set name along with size
pprint.pprint("Final clean JSON File : " + clean_json_data + ", Size : " + str(os.path.getsize(clean_json_data) >> 20) + " MB")

print os.path.basename(__file__) ,"script Ended.Duration: ", round(time()-t0,3), "s"	

