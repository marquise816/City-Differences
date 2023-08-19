import methods as m

# Connect to Local MongoDB
MongoDBLink = "mongodb://localhost:27017"

m.mongoDBClientConnection(MongoDBLink, 'Yelp_Dataset_80,000')

# import 80,000 entries into mongoDB
m.insertJsonData('yelp_academic_dataset_business.json')

# Turn Query result into Pandas DataFrame
originalData = m.queryFields_WithCoord()

originalData['City'] = m.cleanCityNames(originalData['City'], originalData['Latitude'], originalData['Longitude'] )
# Saving to prevent having to clean city names again
originalData.to_csv('Original_Data.csv')
# Describe Data
print('--- Original Data Description ---')
print(originalData['Category'].describe())
print(originalData['Stars'].describe())

m.sleep(1)
# Get visuals of original data
m.histogram(originalData['Category'], 'Categories', 9, 'Original_Data_Hist_Category.png')
m.histogram(originalData['Stars'], 'Stars', 9, 'Original_Data_Hist_Star.png')
m.plotScatter('Stars', 'Category', 'City', originalData, 'Original_Data_Scatter_Star_Category.png')

# Merging By Cities
m.sleep(1)
mergedData = m.mergeByCity(originalData)

# Saving to prevent restarting from Original Data
mergedData.to_csv('Merged_Data.csv')

# Describe Data
m.sleep(1)
print('--- Merged Data Description ---')
print(mergedData['Category_Avg'].describe())
print(mergedData['Stars_Avg'].describe())
print(mergedData['Businesses'].describe())

# Get visuals of Merged data
m.sleep(1)
m.histogram(mergedData['Category'], 'Categories', 9, 'Merged_Data_Hist_Category.png')
m.histogram(mergedData['Stars'], 'Stars', 9, 'Merged_Data_Hist_Star.png')
m.histogram(mergedData['Businesses'], 'Stars', 9, 'Merged_Data_Hist_Businesses.png')
m.plotScatter('Stars', 'Category', 'City', mergedData, 'Merged_Data_Scatter_Star_Category.png')
m.plotScatter('Businesses', 'Category', 'City', mergedData, 'Merged_Data_Scatter_Business_Category.png')
m.plotScatter('Businesses', 'Stars', 'City', mergedData, 'Merged_Data_Scatter_Business_Star.png')


''' Normalization'''

# Clip Normalization
normData = mergedData.copy()
normData['Businesses']= normData['Businesses'].clip(upper=44) # clips 194 entries
normData['Stars_Avg']= normData['Stars_Avg'].clip(lower=3.286, upper=3.921) # clips 382 entries
normData['Category_Avg']= normData['Category_Avg'].clip(lower=3.8, upper=4.74) # clips 384 entries

#MinMax
normData['Businesses'] = m.min_max_normalize(normData['Businesses'])
normData['Stars_Avg'] = m.min_max_normalize(normData['Stars_Avg'])
normData['Category_Avg'] = m.min_max_normalize(normData['Category_Avg'])

# Saving to prevent restarting from Original Data
normData.to_csv('Normalized_Data.csv')

m.sleep(1)
m.histogram(normData['Category_Avg'], 'Categories', 9, 'Norm_Data_Hist_Category_Avg.png')
m.histogram(normData['Stars_Avg'], 'Stars_Avg', 9, 'Norm_Data_Hist_Stars_Avg.png')
m.histogram(normData['Businesses'], 'Business', 9, 'Norm_Data_Hist_Businesses.png')
m.plotScatter('Stars_Avg', 'Category_Avg', 'City', normData, 'Norm_Data_Scatter_Stars_Avg_Category_Avg.png')
m.plotScatter('Businesses', 'Category_Avg', 'City', normData, 'Norm_Data_Scatter_Business_Category_Avg.png')
m.plotScatter('Businesses', 'Stars_Avg', 'City', normData, 'Norm_Data_Scatter_Business_Stars_Avg.png')



m.sleep(1)
''' Original Similarity'''
similarity = {}
cosine = m.cosine_similarity( originalData['Stars'], originalData['Category'])
euc = m.euclidean_similarity( originalData['Stars'], originalData['Category'])
similarity['Star-Category'] = [cosine, euc]
print('                     Cosine          Euclidean')
m.printSim(similarity)

m.sleep(1)
''' Merged Similarity'''
similarity = {}
cosine = m.cosine_similarity( mergedData['Stars_Avg'], mergedData['Category_Avg'])
euc = m.euclidean_similarity( mergedData['Stars_Avg'], mergedData['Category_Avg'])
similarity['Star-Category'] = [cosine, euc]

cosine = m.cosine_similarity( normData['Businesses'], mergedData['Category_Avg'])
euc = m.euclidean_similarity( normData['Businesses'], mergedData['Category_Avg'])
similarity['Business-Category'] = [cosine, euc]

cosine = m.cosine_similarity( mergedData['Businesses'], mergedData['Stars_Avg'])
euc = m.euclidean_similarity( mergedData['Businesses'], mergedData['Stars_Avg'])
similarity['Business-Star'] = [cosine, euc]
print('---Merged Data Similarity---')
print('                     Cosine          Euclidean')
m.printSim(similarity)


m.sleep(1)
''' Normalizaed Similarity'''
similarity = {}
cosine = m.cosine_similarity( normData['Stars_Avg'], normData['Category_Avg'])
euc = m.euclidean_similarity( normData['Stars_Avg'], normData['Category_Avg'])
similarity['Star-Category'] = [cosine, euc]

cosine = m.cosine_similarity( normData['Businesses'], normData['Category_Avg'])
euc = m.euclidean_similarity( normData['Businesses'], normData['Category_Avg'])
similarity['Business-Category'] = [cosine, euc]

cosine = m.cosine_similarity( normData['Businesses'], normData['Stars_Avg'])
euc = m.euclidean_similarity( normData['Businesses'], normData['Stars_Avg'])
similarity['Business-Star'] = [cosine, euc]
print('---Normalized Data Similarity---')
print('                     Cosine          Euclidean')
m.printSim(similarity)