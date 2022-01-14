import numpy as np
import pandas as pd
import csv
import s2sphere
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import itertools
import gmplot
from keras.models import load_model
import tensorflow as tf
from scipy.interpolate import interp1d

path = '/Users/jh/github/freemove/data/tapas_single_coordinates.csv'


def add_cellID(lat, lng):

    r = s2sphere.RegionCoverer()

    r.max_level = 20
    r.min_level = 5
    r.max_cells = 5000

    cellIDs = []
    for i in range(0, len(lat)):
        p = s2sphere.LatLng.from_degrees(float(lat[i]), float(lng[i]))
        c = s2sphere.CellId.from_lat_lng(p)
        cellIDs.append(c.id())

    return cellIDs


def interpolate(x,y):
    #https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points/51515357
    # Linear length on the line
    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    distance = distance/distance[-1]

    fx, fy = interp1d( distance, x ), interp1d( distance, y )

    alpha = np.linspace(0, 1, 500)
    x_regular, y_regular = fx(alpha), fy(alpha)

    return x_regular, y_regular



def split_data(df, split=0.8):

    num_trips = len(df['TRIP_ID|integer'].unique())
    counter = 0
    test = []
    train = []
    startpoints =  []

    for tripID in df['TRIP_ID|integer'].unique():
            counter +=1
            trip = df.loc[df['TRIP_ID|integer'] == tripID]
            X = trip['X']
            Y = trip['Y']
            X, Y = interpolate(X,Y)
            cellID = add_cellID(X,Y)

            if counter < num_trips*split:
                train.append(cellID)
            else:
                startpoints.append(cellID[0])
                test.append(cellID)
    
    return np.array(train), np.array(startpoints)
                

def save_test_set(df, split=0.8):

    num_trips = len(df['TRIP_ID|integer'].unique())
    counter = 0
    test = []

    for tripID in df['TRIP_ID|integer'].unique():
            counter +=1
            trip = df.loc[df['TRIP_ID|integer'] == tripID]
            X, Y = interpolate(trip['X'],trip['Y'])
            if counter < num_trips*split:
                pass
            else:
                for pair in ((X[i], X[i]) for i in range(min(len(X), len(Y)))):
                    test.append(pair)  

    return test

def create_dataset(df, previous=1):
    dataX, dataY = [], []
    for i in range(len(df)-previous-1):
        a = df[i:(i+previous), 0]
        dataX.append(a)
        dataY.append(df[i + previous, 0])
    return np.array(dataX), np.array(dataY)

def generate_data(path):

    output = {'Type':[],'Coordinates':[]}

    df = pd.read_csv(path)
    df = df.sort_values(by=['PERS_ID|integer','TRIP_ID|integer','sequence']) # sort trips
    
    test = save_test_set(df)
    for x in range(1,int(len(test)/500)+1):
        num = 500*x
        coordinates = test[:num]
        output['Type'].append('original')
        output['Coordinates'].append(coordinates)

    train, startpoints = split_data(df)
    
    #normalize and reshape
    scaler = MinMaxScaler(feature_range=(0, 1))
    #test = test.reshape(-1, 1)
    #test = scaler.fit_transform(test)
    train = train.reshape(-1,1)
    train = scaler.fit_transform(train)
    startpoints = startpoints.reshape(-1,1)
    startpoints = scaler.fit_transform(startpoints)


    #Create dataset with lookback
    look_back=2
    trainX, trainY = create_dataset(train, look_back)
    #testX, testY = create_dataset(df_test, look_back)

    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_dim=1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

    #predict synthetic data
    for point in startpoints:
        point = np.array(point)
        point = point.reshape(-1,1)
        predict = model.predict(point)

        trajectory = []
        for i in range(0, 500):
            p = np.reshape(predict, (predict.shape[0], predict.shape[1], 1))
            predict = model.predict(p)
            trajectory.append(scaler.inverse_transform(predict)) #inverse normalization

        s_c_id = list(itertools.chain(*trajectory)) #change to one iterable

        #make list of cellIDs
        cellId = []
        for i in range(0, len(s_c_id)):
            cellId.append(s_c_id[i][0])
        cellId = list(map(int, cellId))

        #get lat and lng from CellIDs
        map_lat = []
        map_lng = []
        for i in range(0, len(s_c_id)):
            ll = str(s2sphere.CellId(cellId[i]).to_lat_lng())
            latlng = ll.split(',', 1)
            lat = latlng[0].split(':', 1)
            map_lat.append(float(lat[1]))
            map_lng.append(float(latlng[1]))

        #add coordinates to output
        coordinates = list(zip(map_lat, map_lng))
        # coordinates = str(coordinates)
        # chars_to_replace = "[](),"
        # def replace(text):
        #     for char in chars_to_replace:
        #         if char in text:
        #             text = text.replace(char, "")
        #     return text
        # coordinates = replace(coordinates)
        output['Type'].append('synthetic')
        output['Coordinates'].append(coordinates)
    

    #save synthetic trajectories to csv
    pred_df = pd.DataFrame.from_dict(output)
    pred_df.to_csv('/Users/jh/github/freemove/data/predictions_both.csv')



generate_data(path)
