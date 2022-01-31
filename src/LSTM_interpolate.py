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
from keras.models import load_model
import tensorflow as tf
from scipy.interpolate import interp1d
from geojson import LineString, Feature, FeatureCollection, dump
import h3
from utils import add_cellID, interpolate, add_h3, shuffle_trips, train_test_split
import random


path = '/Users/jh/github/freemove/data/tapas_single_coordinates.csv'

# def interpolate(x,y, num_points=500):
#     #https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points/51515357
#     # Linear length on the line
#     distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
#     distance = distance/distance[-1]

#     fx, fy = interp1d( distance, x ), interp1d( distance, y )

#     alpha = np.linspace(0, 1, num_points)
#     x_regular, y_regular = fx(alpha), fy(alpha)

#     return x_regular, y_regular

# def add_h3(lat,lng, resolution=9):
#     """
#     Add unique cell Id by Uber H3 for each coordinate pair
    
#     :param lat:         List of Latitudes
#     :param lng:         List of Longitudes
#     :param resolution:  Controls the size of the hexagon and the number of unique indexes (0 is coarsest and 15 is finest resolution), see https://h3geo.org/docs/core-library/restable/
#     :returns:           List with h3 indexes of the coordinates
#     """
#     h3_indexes = []

#     for i in range(0, len(lat)):
#         index = h3.geo_to_h3(lat[i], lng[i], resolution)
#         h3_indexes.append(index)
    
#     #return h3.geo_to_h3(lat, lng, resolution) 
#     return h3_indexes

# def shuffle_trips(df):
#     """
#     Shuffle trips in the dataset so trips of the same user end up in the train and in the test set
#     :param df:      DataFrame sorted by user and trip ID
#     :returns:       DataFrame with sorted trips but shuffled users
#     """

#     df = df.sort_values(by=['TRIP_ID|integer','sequence']) # sort trips by Trip ID and sequence number
#     groups = [df for _, df in df.groupby('TRIP_ID|integer')] # group trips by trip ID
#     random.shuffle(groups) # shuffle trips
#     df = pd.concat(groups).reset_index(drop=True)
#     return df


# def train_test_split(df, train_size=0.8):
#     """
#     Split the DataFrame into a train and test DataFrame with a given train_size

#     :param df:     Original DataFrame containing the whole dataset
#     :returns:      Train and test DataFrame 
#     """

#     unique_trips = df['TRIP_ID|integer'].unique()
#     train_trips, test_trips = np.split(unique_trips, [int(len(unique_trips)*train_size)])
#     train = df.loc[df['TRIP_ID|integer'].isin(train_trips)]
#     test = df.loc[df['TRIP_ID|integer'].isin(test_trips)]
#     return train, test




def split_data(df, tesselation, split=0.9):
    """
    Split dataset in train and test

    :param df:  DataFrame containing the whole dataset
    :split:     Split for train and test
    :returns:   Array of train dataset and array of startpoints of the test dataset
    """
    
    df = shuffle_trips(df)

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
        X, Y = interpolate(X,Y,num_points=500)
        cellID = tesselation(X,Y)

        if counter < num_trips*split:
            train.append(cellID)
        else:
            startpoints.append(cellID[0])
            test.append(cellID)
    
    return np.array(train), np.array(startpoints)


def save_as_geojson(df):

    features = []
    type = df['Type']
    coordinates = df['Coordinates']
    for i in range(len(df)):
        features.append(Feature(geometry=coordinates[i], properties={"Type": type[i]}))

    feature_collection = FeatureCollection(features)
    with open('/Users/jh/github/freemove/data/predictions.geojson', 'w') as f:
        dump(feature_collection, f)

# def save_test_set(df, split=0.8):
#     """
#     returns a list of coordinaates in tuple form
#     """

#     num_trips = len(df['TRIP_ID|integer'].unique())
#     counter = 0
#     test = []
#     for tripID in df['TRIP_ID|integer'].unique():
#             counter +=1
#             trip = df.loc[df['TRIP_ID|integer'] == tripID]
#             X, Y = interpolate(trip['X'],trip['Y'], num_trips=500)
#             if counter < num_trips*split:
#                 pass
#             else:
#                 for pair in ((X[i], Y[i]) for i in range(min(len(X), len(Y)))):
#                     test.append(pair)  

#     return test

def test_coordinates(df):
    """
    Returns a list of coordinates in tuple form

    :param df:      Test DataFrame
    :returns:       List of tuples with X,Y coordinates
    """
    test = []
    for tripID in df['TRIP_ID|integer'].unique():
        trip = df.loc[df['TRIP_ID|integer'] == tripID]
    #TODO
    return test

def generate_output_df(test_coordinates, pred_coordinates):
    
    #divide into trips and add type and append to dataframe
    coordinates = []
    type = []

    for x in range(1,int(len(test_coordinates)/500)+1):
        num = 500*x
        coords = test_coordinates[:num]
        type.append('original')
        coordinates.append(LineString(coords))

    for x in range(1,int(len(pred_coordinates)/500)+1):
        num = 500*x
        coords = pred_coordinates[:num]
        type.append('synthetic')
        coordinates.append(LineString(coords))

    data = pd.DataFrame(columns=['Type', 'Coordinates'])
    data['Type'] = type
    data['Coordinates'] = coordinates
    
    return data

def create_dataset(df, previous=1):
    """
    Create data with a lookback

    :param df:          DataFrame with location data
    :param previous:    Lookback
    :returns:           Tuple of arrays of X and Y data with a lookback  
    """

    dataX, dataY = [], []
    for i in range(len(df)-previous-1):
        a = df[i:(i+previous), 0]
        dataX.append(a)
        dataY.append(df[i + previous, 0])
    return np.array(dataX), np.array(dataY)

def lookback_try(df, previous=2):

    dataX, dataY = [], []
    for traj in df:
        traj = np.array(traj)
        traj = traj.reshape(-1,1)     
        x = []
        y = []
        for i in range(len(traj)-previous):
            a = traj[i:(i+previous), 0]
            x.append(a)
            y.append(traj[i + previous, 0])  
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
 
def unique_values(df):
    df = df.reshape(-1,1)
    df = np.unique(df)
    print('Number of unique values: ', len(df))


def interpolate_tesselation(df, tesselation=add_h3):
    """
    Takes DataFrame and interpolates the X and Y coordinates, converts them into tesselation cell IDs

    :param df:              DataFrame
    :param tesselation:     Function for tesselation
    :returns:               Array with tesselation cells
    """

    data = []
    for tripID in df['TRIP_ID|integer'].unique():
        trip = df.loc[df['TRIP_ID|integer'] == tripID]
        X, Y = trip['X'], trip['Y']
        X, Y = interpolate(X, Y, num_points=500)
        tesselation_cells = tesselation(X,Y)
        print(len(set(tesselation_cells)))
        data.append(tesselation_cells)
    return np.array(data)


def get_startpoints(data):
    startpoints = []
    for trip in data:
        startpoints.append(trip[0])
    return startpoints

def generate_data(path):

    df = pd.read_csv(path)
    df = shuffle_trips(df)
    df_train, df_test = train_test_split(df, train_size=0.8)
    print(len(df_train['TRIP_ID|integer'].unique())==80)

    train = interpolate_tesselation(df_train)
    test = interpolate_tesselation(df_test)
    print(train)
    startpoints = get_startpoints(test)    

    #train, startpoints = split_data(df, tesselation=add_h3)
    #unique_values(train)
    #unique_values(startpoints)
    #print(train.shape)
    #num_trips = train.shape[0]
    #print(num_trips)



    #normalize and reshape
    scaler = MinMaxScaler(feature_range=(0, 1))
    train = train.reshape(-1,1)
    print(train.shape)
    #train = scaler.fit_transform(train)
    startpoints = startpoints.reshape(-1,1)
    #startpoints = scaler.fit_transform(startpoints)


    #Create dataset with lookback
    look_back=4
    #trainX, trainY = lookback_try(train, previous=2)
    trainX, trainY = create_dataset(train, look_back)
    #testX, testY = data_lookback(df_test, look_back)
    #trainX = trainX.reshape(-1,1)
    #trainY = trainY.reshape(-1,1)
    print('trainx',trainX.shape)
    print('train y', trainY.shape)
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    ####train = np.reshape(train, (train.shape[0], train.shape[1], 1))
    print(trainX.shape)
    batchsize = trainX.shape[0]/num_trips
    print(batchsize)



    # create and fit the LSTM network
    model = Sequential()
    ####model.add(LSTM(4, input_dim=1))
    model.add(LSTM(4, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size= 500, verbose=2)
    ####model.fit(train, epochs=2, batch_size=1, verbose=2)


    trajectory = []

    #predict synthetic data
    for point in startpoints:
        point = np.array(point)
        point = point.reshape(-1,1)
        predict = point
        #predict = model.predict(point)

        #trajectory = []
        #trajectory.append(scaler.inverse_transform(predict)) #inverse normalization
        trajectory.append(predict)
        for i in range(0,499):
            p = np.reshape(predict, (predict.shape[0], predict.shape[1], 1))
            predict = model.predict(p)
            trajectory.append(predict)
            #trajectory.append(scaler.inverse_transform(predict)) #inverse normalization

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
    pred_coordinates = list(zip(map_lat, map_lng))

    #TODO make output a list not dict because dict unordered
    #if test is also two lists then merge into df?
    #then save as geojson? or directly put lists into a geodf?

    #coordinates_output = []
    #type_output = ['synthetic'*len(coordinates)]
    #output = {'Type':[],'Coordinates':[]}
    #output['Type'].append('synthetic')
    #output['Coordinates'].append(LineString(coordinates))

    
    test_coordinates = save_test_set(df)
    output_df = generate_output_df(test_coordinates, pred_coordinates)

    #output_df = output_df.append(output, ignore_index=True)
    print(output_df.head())
    #output_df.to_csv('/Users/jh/github/freemove/data/synthetic_data.csv')
    save_as_geojson(output_df)


    #save synthetic trajectories to csv
    #pred_df = pd.DataFrame.from_dict(output)
    #pred_df.to_csv('/Users/jh/github/freemove/data/predictions_both.csv')



generate_data(path)
