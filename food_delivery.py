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
data['distance'] = np.nan

for i in range(len(data)):
    data.loc[i,'distance']=distanceCalculate(data.loc[i, 'Restaurant_latitude'], 
                                        data.loc[i, 'Restaurant_longitude'], 
                                        data.loc[i, 'Delivery_location_latitude'], 
                                        data.loc[i, 'Delivery_location_longitude'])

#Data Exploration 
#finding the relationships between the features

# 1. relationship of distance and the time taken to diliver

figure = px.scatter(data_frame = data, 
                    x="distance",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    trendline="ols", 
                    title = "Relationship Between Distance and Time Taken")
figure.show()

#the cahrt dispayes that consisten that most partners diliver food by 20-30 min
#regardles of the distance

#2. relationship between the dilivry time and diviery partners age

figure = px.scatter(data_frame = data, 
                    x="Delivery_person_Age",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between delivery partner age and Time Taken")
figure.show()

#03.relationship between the dilivery time and the delivery partners ratings

figure = px.scatter(data_frame = data, 
                    x="Delivery_person_Ratings",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time Taken and Ratings")
figure.show()

#04.the relationship of the type of food orderd and vehicle the delivery partner have

fig = px.box(data, 
             x="Type_of_vehicle",
             y="Time_taken(min)", 
             color="Type_of_order")
fig.show()

#for the conclusion the age,distance,rating affect the devlivery time

#train a machine learing model using a LSTM neural network model 


##slitting the data 
#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[["Delivery_person_Age", 
                   "Delivery_person_Ratings", 
                   "distance"]])
y = np.array(data[["Time_taken(min)"]])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

# creating the LSTM neural network model
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

# training the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=9)

print("Food Delivery Time Prediction")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a, b, c]])
print("Predicted Delivery Time in Minutes =",model.predict(features))

model.save("saved_model.keras")  # Save the model i trained
