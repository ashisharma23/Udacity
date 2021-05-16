#!/usr/bin/python

'''
This file contains various modules being imported by source files.
'''
import xml.etree.cElementTree as ET # SAX XML parser
import pprint #Pretty print
import re  # Regular expression
import codecs # encoders and decoders
import json # json encoders and decodes
import collections # specialized container datatypes
import random #Generate pseudo-random numbers
import pymongo # MongoDB library
import os # Miscellaneous operating system interfaces
from time import time
from collections import defaultdict

'''
Input file details
'''
datadir = "data"  # folder name where osm xml files is located
osmfile = "san-jose_california.osm" # name of original osm file downloaded 
sanjose_data = os.path.join(datadir, osmfile) # reference to original osm file under data folder
sample_json_data = sanjose_data[:-4]+"_tag_sample.json" # name of sample json file generated for manaul review
clean_json_data = sanjose_data[:-4]+ "_clean.json" # name of final json file created for importing into mongoDB.
