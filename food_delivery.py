import pandas as pd   #used pandas for data manipulation and alanysis
import numpy as np  #i use it for numerical calculations and arrays
import plotly.express as px   #used for visualization of datasets

#read the dataset
data = pd.read_csv("deliverytime.txt")
#view the 1st few rows
print(data.head())
#see what the columns in the data set consists
data.info()
#look the dataset to see if there are null values here 
data.isnull().sum()

#calculate the distance between 2 latitudes and longitued using haversine formulea

##set the earth radius 
R=6371

##convert degrees to radians
def deg_to_rad(degrees):
    return degrees * (np.pi/180)

#calculate the ditance of 2 points
def distanceCalculate(lat1,lon1,lat2,lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon=deg_to_rad(lon2-lon1)
    a=np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2))* np.sin(d_lon/2)**2
    c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    return R *c

#calculate distance between each pair of points
data['distance'] = np.nam

for i in range(len(data)):
    data.loc[i,'distance']=distanceCalculate(data.loc[i, 'Restaurant_latitude'], 
                                        data.loc[i, 'Restaurant_longitude'], 
                                        data.loc[i, 'Delivery_location_latitude'], 
                                        data.loc[i, 'Delivery_location_longitude'])

#Data Exploration 