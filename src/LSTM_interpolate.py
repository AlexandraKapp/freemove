from re import T
import numpy as np
import pandas as pd
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
from geojson import LineString, Feature, FeatureCollection, dump
import h3
from utils import add_h3, add_s2sphere, predict_synthetic_new, reverse_h3, shuffle_trips, train_test_split, interpolate_tesselation, final_lookback, get_startpoints


path = '/Users/jh/github/freemove/data/tapas_single_coordinates.csv'


def save_as_geojson(df):

    features = []
    type = df['Type']
    coordinates = df['Coordinates']
    for i in range(len(df)):
        features.append(Feature(geometry=coordinates[i], properties={"Type": type[i]}))

    feature_collection = FeatureCollection(features)
    with open('/Users/jh/github/freemove/data/predictions.geojson', 'w') as f:
        dump(feature_collection, f)

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

 
# def unique_values(df):
#     df = df.reshape(-1,1)
#     df = np.unique(df)
#     print('Number of unique values: ', len(df))
    
def predict_synthetic(model, startpoints, lookback):
    
    while startpoints.shape[1] < 500:
        predict = model.predict(startpoints[:, :lookback, :])
        predict = np.reshape(predict, (predict.shape[0], predict.shape[1], 1))
        startpoints = np.hstack((startpoints, predict))

    # reverse normalization
    # reverse tesselation

    return startpoints


def generate_data(path):

    df = pd.read_csv(path)
    df = shuffle_trips(df)
    df_train, df_test = train_test_split(df, train_size=0.8)
    

    all_data = interpolate_tesselation(df, tesselation=add_h3)
    train = interpolate_tesselation(df_train, tesselation=add_h3)
    test = interpolate_tesselation(df_test, tesselation=add_h3)
    startpoints = get_startpoints(test)  


    #normalize and reshape
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_data = all_data.reshape(-1,1)
    scaler.fit(all_data)

    print('Train shape before reshape: ', train.shape)
    train = train.reshape(-1,1)
    train = scaler.transform(train)
    print('Train shape after normalisation: ', train.shape)
    train = np.reshape(train, (80, 500))
    print('Train shape after reshape: ', train.shape)
    startpoints_shape = startpoints.shape
    startpoints = startpoints.reshape(-1,1)
    startpoints = scaler.transform(startpoints)
    startpoints = np.reshape(startpoints, startpoints_shape)
    print('Startpoints shape after normalisation: ', startpoints.shape)


    lookback = 3
    #trainX, trainY = transform_data(train)
    trainX, trainY = final_lookback(train, previous=lookback)


    #Create dataset with lookback
    #look_back=4
    #trainX, trainY = lookback_try(train, previous=2)
    #trainX, trainY = create_dataset(train, look_back)
    #testX, testY = data_lookback(df_test, look_back)
    #trainX = trainX.reshape(-1,1)
    #trainY = trainY.reshape(-1,1)
    print('trainx',trainX.shape)
    print('train y', trainY.shape)
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    ####train = np.reshape(train, (train.shape[0], train.shape[1], 1))
    
    print('Train shape: ', trainX.shape)
    #batchsize = trainX.shape[0]/num_trips
    #print(batchsize)



    # create and fit the LSTM network
    model = Sequential()
    ####model.add(LSTM(4, input_dim=1))
    model.add(LSTM(10, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size= 500, verbose=2)
    ####model.fit(train, epochs=2, batch_size=1, verbose=2)


    #trajectory = []
    #startpoints = np.reshape(startpoints, (startpoints.shape[0], startpoints.shape[1], 1))
    print(startpoints.shape)
    #new = startpoints[:, :lookback, :]
    #print(new.shape)
    #trajectory.append(startpoints)

    coord = predict_synthetic_new(model, startpoints, lookback)
    coord = np.reshape(coord, (coord.shape[0], coord.shape[1]))
    #coord = scaler.inverse_transform(coord)
    print(np.amin(test))
    print(np.amin(coord))
    print(coord.shape)
    # predict = model.predict(startpoints[:, :lookback, :])
    # predict = np.reshape(predict, (predict.shape[0], predict.shape[1], 1))
    # new =np.hstack((startpoints, predict))
    # print(new.shape)
    #x = trajectory.append(predict)

    # #predict synthetic data
    # for point in startpoints:
    #     point = np.array(point)
    #     #point = point.reshape(-1,1)
    #     predict = point
    #     #predict = model.predict(point)

    #     #trajectory = []
    #     #trajectory.append(scaler.inverse_transform(predict)) #inverse normalization
    #     trajectory.append(predict)
    #     for i in range(0,499):
    #         predict = predict.reshape(-1,1)
    #         p = np.reshape(predict, (predict.shape[0], predict.shape[1], 1))
    #         predict = model.predict(p)
    #         trajectory.append(predict)
    #         trajectory.append(scaler.inverse_transform(predict)) #inverse normalization

    # s_c_id = list(itertools.chain(*trajectory)) #change to one iterable

    # #make list of cellIDs
    # cellId = []
    # for i in range(0, len(s_c_id)):
    #     cellId.append(s_c_id[i][0])
    # cellId = list(map(int, cellId))

    # #get lat and lng from CellIDs
    # map_lat = []
    # map_lng = []
    # for i in range(0, len(s_c_id)):
    #     ll = str(s2sphere.CellId(cellId[i]).to_lat_lng())
    #     latlng = ll.split(',', 1)
    #     lat = latlng[0].split(':', 1)
    #     map_lat.append(float(lat[1]))
    #     map_lng.append(float(latlng[1]))

    # #add coordinates to output
    # pred_coordinates = list(zip(map_lat, map_lng))

    # #TODO make output a list not dict because dict unordered
    # #if test is also two lists then merge into df?
    # #then save as geojson? or directly put lists into a geodf?

    # #coordinates_output = []
    # #type_output = ['synthetic'*len(coordinates)]
    # #output = {'Type':[],'Coordinates':[]}
    # #output['Type'].append('synthetic')
    # #output['Coordinates'].append(LineString(coordinates))

    
    # test_coordinates = save_test_set(df)
    # output_df = generate_output_df(test_coordinates, pred_coordinates)

    # #output_df = output_df.append(output, ignore_index=True)
    # print(output_df.head())
    # #output_df.to_csv('/Users/jh/github/freemove/data/synthetic_data.csv')
    # save_as_geojson(output_df)


    # #save synthetic trajectories to csv
    # #pred_df = pd.DataFrame.from_dict(output)
    # #pred_df.to_csv('/Users/jh/github/freemove/data/predictions_both.csv')



generate_data(path)
