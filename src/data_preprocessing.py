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


def add_cellID(df):

    #convert to s2sphere and add cellid
    
    r = s2sphere.RegionCoverer()

    #Do I need this region to get the covering? Only for plotting but it doesnt work
    maxX,maxY,minX,minY = df['X'].max(),df['Y'].max(),df['X'].min(),df['Y'].min()

    region = s2sphere.LatLngRect(
        s2sphere.LatLng.from_degrees(maxX, maxY),
        s2sphere.LatLng.from_degrees(minX, minY))

    #These parameters do not influence the number of unique CellIDs? Why?
    r.max_level = 30
    r.min_level = 20
    r.max_cells = 5000
    covering = r.get_covering(region)

    lat = df.loc[:, 'X']
    lng = df.loc[:, 'Y']
    #convert latlong to cellIDs
    cellIDs = []
    for i in range(0, len(lat)):
        p = s2sphere.LatLng.from_degrees(float(lat[i]), float(lng[i]))
        c = s2sphere.CellId.from_lat_lng(p)
        cellIDs.append(c.id())


    df['CellID'] = np.array(cellIDs)
    #df['CellID'] = pd.to_numeric(df['CellID'], downcast="float")

    return df


def interpolate(df):

    # Linear length on the line
    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    distance = distance/distance[-1]

    fx, fy = interp1d( distance, x ), interp1d( distance, y )

    alpha = np.linspace(0, 1, 15)
    x_regular, y_regular = fx(alpha), fy(alpha)



def normalize_df(data):
    # normalize the dataset after train-test-split!!!!!
    # quote: https://towardsdatascience.com/one-step-predictions-with-lstm-forecasting-hotel-revenues-c9ef0d3ef2df
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(data)
    return df

def drop_columns_except(df,columns):
    return df[columns]

def train_test_split(df,split=0.95):
    #TODO dont split users

    # split into train and test sets
    split = int(len(df) * split)
    df_test = df.iloc[split:,:]
    df_test = df_test.reset_index(drop=True)
    df_train = df.iloc[:split,:]
    df_train = df_train.reset_index(drop=True)

    return df_train, df_test


def new_train_test(df, split=0.8):

    num_trips = len(df['TRIP_ID|integer'].unique())

    #pad_train = []
    #pad_test = []
    #startpoints = []
    #counter = 0

    # #train test split
    # for tripID in df['TRIP_ID|integer'].unique():
    #     counter +=1
    #     if counter < num_trips*split:
    #         pad_train.append(df.loc[df['TRIP_ID|integer'] == tripID])
    #     else:
    #         pad_test.append(df.loc[df['TRIP_ID|integer'] == tripID])

    pad_train = df[df['TRIP_ID|integer'] <= num_trips*split]
    pad_test = df[df['TRIP_ID|integer'] > num_trips*split]

    startpoints = []
    for tripID in pad_test['TRIP_ID|integer'].unique():
        x = pad_test.loc[df['TRIP_ID|integer'] == tripID]
        startpoints.append(x['CellID'].iloc[0])

    startpoints = np.array(startpoints)
    pad_train = np.array(pad_train['CellID'])
    pad_test = np.array(pad_test['CellID'])
    
   # pad_train = np.concatenate(pad_train, axis=0 )
   # pad_test = np.concatenate(pad_test, axis=0 )

    scaler = MinMaxScaler(feature_range=(1,2)) #normalize train and test seperately

    #normalize
    pad_train = pad_train.reshape(-1, 1)
    pad_test = pad_test.reshape(-1, 1)
    startpoints = startpoints.reshape(-1, 1)
    pad_train = scaler.fit_transform(pad_train)
    pad_test = scaler.fit_transform(pad_test)
    #startpoints = startpoints.fit_transform(startpoints)

    #post padding with zeros
    pad_train = tf.keras.preprocessing.sequence.pad_sequences(pad_train, padding='post')
    pad_test = tf.keras.preprocessing.sequence.pad_sequences(pad_test, padding='post')

    return pad_train, pad_test, startpoints

def train_test_split_padding(df,split=0.80):

    max_seq = df['sequence'].max()
    num_trips = len(df['TRIP_ID|integer'].unique())

    pad_train = []
    pad_test = []
    startpoints = []
    counter = 0

    for tripID in df['TRIP_ID|integer'].unique():
        counter +=1
        if counter < num_trips*split:
            x = df.loc[df['TRIP_ID|integer'] == tripID]
            cellids = np.array(x['CellID'])
            listofzeros = [0] * (max_seq-len(cellids))
            cellids = np.append(cellids,listofzeros)
            pad_train.append(cellids)
        else:
            x = df.loc[df['TRIP_ID|integer'] == tripID]
            cellids = np.array(x['CellID'])
            startpoints.append(cellids[0])
            listofzeros = [0] * (max_seq-len(cellids))
            cellids = np.append(cellids,listofzeros)
            pad_test.append(cellids)

    pad_train = np.concatenate(pad_train, axis=0 )
    pad_test = np.concatenate(pad_test, axis=0 )
    startpoints = np.array(startpoints)
    #startpoints = np.reshape(startpoints, (startpoints.shape[0], 1, 1))
    return pad_train, pad_test, startpoints


def create_dataset(df, previous=1):
    dataX, dataY = [], []
    for i in range(len(df)-previous-1):
        a = df[i:(i+previous), 0]
        dataX.append(a)
        dataY.append(df[i + previous, 0])
    return np.array(dataX), np.array(dataY)

def generate_data(path):

    df = pd.read_csv(path)
    df = df.sort_values(by=['PERS_ID|integer','TRIP_ID|integer','sequence']) # sort trips
    df = add_cellID(df) # add s2sphere cell IDs
    df = df[['CellID', 'X', 'Y', 'TRIP_ID|integer', 'sequence', 'PERS_ID|integer']] #drop columns except for
    #df_train, df_test = train_test_split(df)
    max_seq = df['sequence'].max()
    #df_train, df_test, startpoints = train_test_split_padding(df)
    
    df_train, df_test, startpoints = new_train_test(df)

    #scaler = MinMaxScaler(feature_range=(0, 1)) #normalize train and test seperately

    #df_test = np.array(df_test['CellID'])
    df_test = df_test.reshape(-1, 1)
    df_test = scaler.fit_transform(df_test)

    #df_train = np.array(df_train['CellID'])
    df_train = df_train.reshape(-1,1)
    df_train = scaler.fit_transform(df_train)

    startpoints = startpoints.reshape(-1,1)
    startpoints = scaler.fit_transform(startpoints)
    #startpoints = np.reshape(startpoints, (startpoints.shape[0], startpoints[1], 1))

    #unique number of Cell IDs
    (unique, counts) = np.unique(df['CellID'], return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(f"unique: {len(unique)}")

    #Create dataset with lookback
    look_back=2
    trainX, trainY = create_dataset(df_train, look_back)
    #testX, testY = create_dataset(df_test, look_back)

    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    #testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


    # create and fit the LSTM network
    # TODO Problem with the batchsize and the length of the input
    model = Sequential()
    model.add(LSTM(4, input_dim=1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)


    #model.save('traffic_gen01.h5')  # creates a HDF5 file 'my_model.h5'

    # serialize model to JSON
    # model_json = model.to_json()
    # with open("traffic_gen01.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("traffic_gen.h5")
    # print("Saved model to disk")


    #what does this predict do?
    #predict = model.predict(testX)

    predict = model.predict(startpoints)

    trajectory = []
    #what is this range?
    for i in range(0, max_seq):
        p = np.reshape(predict, (predict.shape[0], predict.shape[1], 1))
        #Why do we predict the already predicted values?
        predict = model.predict(p)
        trajectory.append(scaler.inverse_transform(predict)) #inverse normalization

    s_c_id = list(itertools.chain(*trajectory)) #What does this do?

    #What happens here with the CellIDs?
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


    #save synthetic trajectories to csv
    lat_lng = {'X':map_lat,'Y':map_lng, 'TRAFFIC_MODE|text': 'Synthetic'}
    pred_df = pd.DataFrame(lat_lng)
    pred_df.to_csv('/Users/jh/github/freemove/data/predictions.csv')

    #plot synthetic trajectory
    plt.plot(map_lat, map_lng)
    plt.show()

    #gmap = gmplot.GoogleMapPlotter(46.519962, 6.633597, 16)
    #gmap = gmplot.GoogleMapPlotter(6.761881, 45.112042, 16)
    #gmap.plot(map_lat, map_lng, '#000000', edge_width=20)
    #gmap.scatter(map_lat, map_lng, '#000000', edge_width=20)
    #gmap.draw("map001.html")


generate_data(path)

"""
    look_back = 5
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    print(f"TrainX: {trainX[0:10]}")
    print(f"TrainY: {trainY[0:10]}")
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    print(f"TrainX: {trainX[0:10]}")
    print(f"TrainY: {trainY[0:10]}")
    return trainX, trainY, testX, testY
"""