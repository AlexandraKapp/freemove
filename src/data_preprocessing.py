import numpy as np
import pandas as pd
import csv
import s2sphere
from sklearn.preprocessing import MinMaxScaler


path = '/Users/jh/github/freemove/data/tapas_single_coordinates.csv'

def load_dataset(path):
    return pd.read_csv(path)

def sort_dataset(df):
    #sort by user, trip, timestep
    return df.sort_values(by=['PERS_ID|integer','TRIP_ID|integer','sequence'])

def add_cellID(df):
    #convert to s2sphere and add cellid
    r = s2sphere.RegionCoverer()
    coverer = s2sphere.RegionCoverer()
    #coverer.set_min_level(8)
    #coverer.set_max_level(15)
    #coverer.set_max_cells(500)
    lat = df.loc[:, 'X']
    lng = df.loc[:, 'Y']
    #convert latlong to cellIDs
    cellIDs = []
    for i in range(0, len(lat)):
        p = s2sphere.LatLng.from_degrees(float(lat[i]), float(lng[i]))
        c = s2sphere.CellId.from_lat_lng(p)
        cellIDs.append(c.id())

    df['CellID'] = np.array(cellIDs)
    df['CellID'] = pd.to_numeric(df['CellID'], downcast="float")

    return df


def normalize_df(data):
    # normalize the dataset after train-test-split!!!!!
    # quote: https://towardsdatascience.com/one-step-predictions-with-lstm-forecasting-hotel-revenues-c9ef0d3ef2df
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(data)
    return df

def drop_columns_except(df,columns):
    return df[columns]

def train_test_split(df,split=0.8):
    #TODO dont split users

    # split into train and test sets
    split = int(len(df) * split)
    df_test = df.iloc[:split,:]
    df_test = df_test.reset_index(drop=True)
    df_train = df.iloc[split:,:]
    df_train = df_train.reset_index(drop=True)

    return df_train, df_test


def create_dataset(df, previous=1):
    dataX, dataY = [], []
    for i in range(len(df)-previous-1):
        a = df[i:(i+previous), 0]
        dataX.append(a)
        dataY.append(df[i + previous, 0])
    return np.array(dataX), np.array(dataY)

def generate_data():

    df = load_dataset(path)
    df = sort_dataset(df)
    df = add_cellID(df)
    #df = drop_columns_except(df, ['CellID', 'X', 'Y', 'TRIP_ID|integer', 'sequence', 'PERS_ID|integer'])
    df_train, df_test = train_test_split(df)
    test = np.array(df['CellID'])
    test = test.reshape(-1, 1)
    train = np.array(df['CellID'])
    (unique, counts) = np.unique(test, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    print(f"frequencies: {frequencies}")
    print(f"unique: {len(unique)}")

generate_data()
"""
    train = train.reshape(-1,1)
    print(train[0:10])
    #train = normalize_df(train)
    #test = normalize_df(test)
    print(test[0:10])

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
