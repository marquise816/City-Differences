from json import loads
from unidecode import unidecode # Converts foreign characters into US Keyboard Charachters
from geopy.geocoders import Nominatim # Uses coordinates to locate address
from matplotlib.pyplot import subplots, clf, xlabel, savefig, ioff, legend
from time import sleep
import seaborn as sns
import pandas as pd
import numpy as np
import pymongo
import re
import geopy.adapters 

### MONGODB Script
# Create a connection to the local Mongo Database
def mongoDBClientConnection( conn_strPar, collectionName):
    global client,db, originalCollection
    
    # set a 5-second connection timeout
    client = pymongo.MongoClient(conn_strPar, serverSelectionTimeoutMS=5000)
    # Check if connection to server is successful
    try: 
        print('Connected to Server')
        print(client.list_database_names())
    except Exception:
        print("Unable to connect to the server.")
        return  
    # Connect to database and collection
    db = client['CIS492']
    originalCollection = db[collectionName]

# Insert Json File  entries into the Mongo Database
def insertJsonData(data):
    with open (data, 'r', encoding='utf-8') as jFile:
        for idx,line in enumerate(jFile):
            if idx == 80000: # Limit it to 80,000 entries
                break
            entry = loads(line)
            originalCollection.insert_one(entry)
    jFile.close()
    print('Data Insertion Complete!')

def changeCollection(name):
    originalCollection = db[name]

def insertToDatabase(df):
    originalCollection.insert_many(df.to_dict('records'))

# Uses the connection with the database to querry some fields  
def queryFields_WithCoord():
    # Gets all entries but only certain fields
    query = originalCollection.find({}, {'_id':0,'name':1,'state':1, 'city':1, 'categories':1, 'stars':1, 'latitude':1, 'longitude':1 })
    d = {"Name":[], "City":[], "State":[], "Stars":[], "Category": [], 'Latitude':[], 'Longitude':[] }
    for entry in query:
        d['Name'].append(entry['name'])
        d['State'].append(entry['state'])
        d['City'].append(removeSymbols(entry['city'])) 
        d['Stars'].append(entry['stars'])
        d['Category'].append(getLength(entry['categories'], ', ')) # Only taking the length of the categories list
        d['Latitude'].append(entry['latitude'])
        d['Longitude'].append(entry['longitude'])
    return pd.DataFrame(d)

# Takes at least 30 minutes to complete
# It takes pandas columns: (City Column, Latitude Column, Longitude Column)
# Returns list of the proper city names
def cleanCityNames(lst, lst2, lst3): # 
    cityNameRep = {}  # A Dictionary to hold the original city names and actual city name
    cityNameList = [] # Contains the changed city names
    for x,(curCity, lat, lon) in enumerate(zip(lst, lst2, lst3)):
        # Check if current city is in dictionary as a key
        if curCity in cityNameRep.keys(): 
            cityNameList.append(cityNameRep[curCity]) 
        else:
                # Check if Empty because of worry about reaching a geopy's limitations or exceptions
            if not curCity == '':
                c = getCityName(lat, lon, curCity)
                c = unidecode(c) 
                cityNameRep[curCity] = c
                cityNameList.append(c)
            else: 
                c = getCityName(lat, lon,'Unknown City')
                c = unidecode(c)
                cityNameList.append(c) 
    return cityNameList

# Merges the business by the city they reside within.
def mergeByCity(df):
    # create dictionary that contains the list of each category. 
    # The Keys are dataframe columns and the list is the entries of the column
    d = {"City":[],'State':[], "Businesses":[], "Stars_Avg":[], "Category_Avg": [], "Stars":[], "Category": []}
    
    # Merging by City
    for idx, (city, stars, category, state) in enumerate(zip(df['City'], df['Stars'], df['Category'], df['State'])):
        # Check if city is already in the city list
        if city in d['City']:
            index =  d['City'].index(city)      # Get  index within city list
            # Sum the entry columns to the dictionaries list
            d['Businesses'][index] += 1         # Add 1 
            d['Stars'][index] += stars          # Add Stars
            d['Category'][index] += category    # Add Category
        else:
            # appends Entry Column values into the dictionaries lists
            d['City'].append(city)
            d['Stars'].append(stars)
            d['Category'].append(category)
            d['State'].append(state)
            d['Businesses'].append(1)        
    
    # get average
    d["Stars_Avg"] = [i / j for i, j in zip(d["Stars"], d['Businesses'])]
    d["Category_Avg"] = [i / j for i, j in zip(d["Category"], d['Businesses'])]
    
    return pd.DataFrame(d)

# Takes coordinates and returns city name
# Can take 1-16 seconds to complete
def getCityName(lat,long, city):
    loc = Nominatim(user_agent='CIS_Project', timeout=None)
    point = (lat, long) #get coordinates
    if(loc.reverse(point)): # Check if nomination can fing location
        for tries in range(0,3):
            sleep(1)
            try:  
                location = loc.reverse(point) # locate by Coordinates
                address = location.raw['address'] # Get Address dictionary   
            except (geopy.adapters.AdapterHTTPError): # If there is an HTTP Error wait and try again
                print('HttPError') 
                sleep(2)
                continue
            except (AttributeError, KeyError, ValueError): # If any other error return original city name
                return unidecode(city) 
            
            locCity = address.get('city', '') # Get the city name from the dictionary
            if locCity == '': # If the location does not have a city, then return original city name
                # get city name from locate by geocode address
                return unidecode(city)
              
            return unidecode(locCity) # Returns city name found
    else:
        return unidecode(city) # return original city name

# Removes special characters such as @#$%
def removeSymbols(str):
    str = unidecode(str)
    return re.sub(r"[^a-zA-Z0-9 ]", " ", str)

# Splits sstring by delimiter and returns the length of the list
def getLength(txt, delimiter):
    if txt == '' or txt == None:
        return 0
    return len(txt.split(delimiter))

# Uses min-max normalization
def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())


''' Similarity Measures '''

# Retruns the cosine similarity measure
def cosine_similarity(list1, list2):
    return np.dot(list1,list2)/(np.linalg.norm(list1)*np.linalg.norm(list2))

# Retruns the Euclidean Distance
def euclidean_similarity(list1, list2):
    return np.linalg.norm(list1 - list2)



''' Plotting methods '''

def plotScatter(xval, yval,coler, df, fname):
    ioff()
    clf()
    sns.scatterplot(x=xval, y=yval, hue= coler, data=df)
    legend().remove()
    savefig(fname)
    
def histogram(col, label, bin, fname):
    ioff()
    clf()
    d= col.value_counts().to_dict()
    fig, ax = subplots(1, 1)
    ax.hist(col, bins=bin)
    xlabel(label)
    savefig(fname)

def printSim(dictionary):
    for k in dictionary:
        print(f'{k}: {dictionary[k]}')