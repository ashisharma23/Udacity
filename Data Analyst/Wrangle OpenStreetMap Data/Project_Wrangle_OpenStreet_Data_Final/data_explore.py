'''
data_explore.py
        - Contains functions to validate, audit, process and update
          phone number value , phone tag
Created on : 1/25/2018
****************************************************************************
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pprint

from pymongo import MongoClient


# Database connection
client = MongoClient()
#Database
db = client['osm']
# Collection
sanjose_data = db['sanjose']

# Data Overview
print "Number of documents = ", sanjose_data.count()
print "Number of Nodes = ", sanjose_data.find({"type":"node"}).count()
print "Number of Ways = ", sanjose_data.find({"type":"way"}).count()
print "Number of unique users = ", len(sanjose_data.distinct("created.user"))

'''
This function creates barplot using dictionary as input.
'''
def show_barplot(dict_var,x_label,y_label, title):
	df = pd.DataFrame()
	df[y_label] = dict_var.keys()
	df[x_label] = dict_var.values()
	sns.barplot(x = x_label, y = y_label, data=df)
	ax = plt.axes()
	ax.set_title(title)
	plt.show()

#Step1  - Identify Top Contributing Users
topUsers =  sanjose_data.aggregate([
        {"$match": {'created.user': {"$exists": 1} }},
            {"$group": {'_id': '$created.user', 'count': {"$sum": 1}}},
             {"$sort": {'count': -1}},
         {"$limit": 10}
])

topUsersDict = {}
print "Top Contributing Users:"
for user in topUsers:
    print "Id = " , user["_id"] , "\tCount = ", user["count"]
    topUsersDict[user["_id"]] = user["count"]

show_barplot(topUsersDict,"Count", "Users","Top Contributing Users") 

'''
Top contributing users andygol and nmixter have more than 280K entries each,
however lot of entries are under same changeset
A changeset is a collection of related changes (new objects, object modifications, or object deletions) applied together to OSM data. 
'''
topUsers =  sanjose_data.aggregate([
       	 {"$match": {'created.user': {"$exists": 1} }},
	 {"$group": {
		"_id" : "$created.user",
		"mset" : {
			"$addToSet" : "$created.changeset"
			 }
 		  } },
	 {"$unwind" : "$mset" },
	 {"$group": {'_id': '$_id', 'count': {"$sum": 1}}},
         {"$sort": {'count': -1}},
         {"$limit": 10}
])

topUsersDict = {}
print "Top Contributing Users:"
for user in topUsers:
    print "Id = " , user["_id"] , "\tCount = ", user["count"]
    topUsersDict[user["_id"]] = user["count"]

show_barplot(topUsersDict,"Count", "Users","Top Contributing Users")


#Step2 - Top amenties in the cities
topAmenties  = sanjose_data.aggregate([
     {'$match': {'amenity': {'$exists': 1} }},
     {'$group': {'_id': '$amenity', 'count': {'$sum': 1}}},
     {'$sort': {'count': -1}},
     {'$limit': 10}
])
topAmenDict = {}
for amen in topAmenties:
    print "Id = " , amen["_id"] , "\tCount = ", amen["count"]
    topAmenDict[amen["_id"]] = amen["count"]

show_barplot(topAmenDict,"Count", "Amenity","Top Amenities in Cities")

'''
Step3 -Restaurants and fast food as they together are 2nd top most amenity after parking.
However, there is a huge difference between count of restaurants when compared to fast foods
This is as per visual analysis from below data, where there was no restaurants in top ten
topFoodChains  = sanjose_data.aggregate([ 
	{'$match': 
		{ "$and" : [ 
            		{ 'amenity': {'$exists': 1}} , 
	    		{ '$or' : 
				[ 
					{'amenity': 'restaurant'},
					{'amenity': 'fast_food'},
				]
			}
                    	]
            	}},
     {'$group': {'_id': '$name', 'count': {'$sum': 1}}}, 
     {'$sort': {'count': -1}}, 
     {'$limit': 10} 
])

topFoodChainDict = {}
for fc in topFoodChains:
    print "Id = " , fc["_id"] , "\tCount = ", fc["count"]
    if fc['_id'] != 'None':
        topFoodChainDict[fc['_id']]=fc["count"]

show_barplot(topFoodChainDict,"Count", "Food Chains","Top FoodChains in Cities")
'''

# Step 4 - Lets find top restaurants in the city

print "Total restaurants in City is " ,sanjose_data.find( {'amenity':'restaurant'}).count()

topRestaurants  = sanjose_data.aggregate([ 
        {'$match': 
                { "$and" : [ 
                        { 'amenity': {'$exists': 1}} , 
                        {'amenity': 'restaurant'},
                                ]
                }},
     {'$group': {'_id': '$name', 'count': {'$sum': 1}}}, 
     {'$sort': {'count': -1}}, 
     {'$limit': 10} 
])

topRestaurantsDict = {}
for fc in topRestaurants:
    print "Id = " , fc["_id"] , "\tCount = ", fc["count"]
    if fc['_id'] != 'None':
        topRestaurantsDict[fc['_id']]=fc["count"]

show_barplot(topRestaurantsDict,"Count", "Restaurants","Top Restaurants in Cities")


#Step5 - Lets find top fast food in the city

print "Total Fast foods in City is " ,sanjose_data.find( {'amenity':'fast_food'}).count()
topFastFood  = sanjose_data.aggregate([ 
        {'$match': 
                { "$and" : [ 
                        { 'amenity': {'$exists': 1}} , 
                                   {'amenity': 'fast_food'},
                        ]
                }},
     {'$group': {'_id': '$name', 'count': {'$sum': 1}}}, 
     {'$sort': {'count': -1}}, 
     {'$limit': 10} 
])

topFastFoodDict = {}
for fc in topFastFood:
    print "Id = " , fc["_id"] , "\tCount = ", fc["count"]
    if fc['_id'] != 'None':
        topFastFoodDict[fc['_id']]=fc["count"]

show_barplot(topFastFoodDict,"Count", "Fast Food Chains","Top Fast Food in Cities")


#Step6 - Lets find top cafes in city
print "Total cafes in City is " ,sanjose_data.find( {'amenity':'cafe'}).count()
topCafes  = sanjose_data.aggregate([
     {'$match': 
          { "$and" : [
            { 'amenity': {'$exists': 1}} ,
            {'amenity': 'cafe'}
                    ]
            }},
     {'$group': {'_id': '$name', 'count': {'$sum': 1}}},
     {'$sort': {'count': -1}},
     {'$limit': 10}
])
topCafesDict = dict()
for cafe in topCafes:
    print "Id = " , cafe["_id"] , "\tCount = ", cafe["count"]
    if cafe["_id"] !=  'None':	
    	topCafesDict[cafe["_id"]] = cafe["count"]

show_barplot(topCafesDict,"Count", "Cafe","Top Cafe in Cities")


#Step7 -Next, lets find top cuisine amongst restaurants and fast food chains.
topCuisines  = sanjose_data.aggregate([
     {'$match': 
          { "$and" : [
            { 'amenity': {'$exists': 1}} ,
            { "$or" : [
                {'amenity': 'restaurant'},
                {'amenity': 'fast_food'}
            ]},
                           ]
            }},
     {"$unwind" : "$cuisine"},	 # cuisine is a list which can contains multiple items.
     {'$group': {'_id': '$cuisine', 'count': {'$sum': 1}}},
     {'$sort': {'count': -1}},
     {'$limit': 10}
])

topCuisinesDict = dict()

for cus in topCuisines:
    print "Id = " , cus["_id"] , "\tCount = ", cus["count"]
    topCuisinesDict[cus["_id"]] = cus["count"]

	
show_barplot(topCuisinesDict,"Count", "Cuisine","Top Cuisines in Cities")


#Step8 - Lets switch over to sports, to find the trend.
topSports  = sanjose_data.aggregate([
     {'$match': { 'sport': {'$exists': 1}} },
     {'$group': {'_id': '$sport', 'count': {'$sum': 1}}},
     {'$sort': {'count': -1}},
     {'$limit': 10}
])

topSportsDict = dict()
for sp in topSports:
    print "Id = " , sp["_id"] , "\tCount = ", sp["count"]
    topSportsDict[sp["_id"]] = sp["count"]

show_barplot(topSportsDict,"Count", "Sport","Top Sports in Cities")

#Step9 - Top most referenced nodes
topNodes = sanjose_data.aggregate([{'$unwind': '$node_refs'}, \
                                            {'$group': {'_id': '$node_refs', \
                                                        'count': {'$sum': 1}}}, \
                                            {'$sort': {'count': -1}}, \
                                            {'$limit': 10}],
                                            allowDiskUse =  True)

for node in topNodes:
    print "Count = ",node['count']	
    pprint.pprint(sanjose_data.find({'id': node['_id']})[0])

