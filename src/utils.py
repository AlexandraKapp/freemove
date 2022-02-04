from geojson_length import calculate_distance, Unit
import numpy as np
import yaml
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import s2sphere
from geojson import LineString, Feature, FeatureCollection, dump
import random
import h3


def repeat_and_collate(classify_fn, **args):
    """
    Repeats classification function call with provided command-line arguments, collects results and prints mean and std.

    :param classify_fn: classification function reference
    :param args: keyword-argument dictionary of command-line parameters provided by argparse
    """
    results = [classify_fn(**args) for _ in range(args['repetitions'])]

    if len(results) > 1:
        print('\nResults of experiment:')
        for i,iteration in enumerate(results):
            print(f'Run {i+1:02d} balanced accuracy:\t{round(results[i], 5)}')
        print(f'Average balanced accuracy:\t{round(np.mean(results), 5)} (\u00B1 {round(np.std(results), 5)})\n')


def print_dataset_info(dataset):
    """
    Prints formatted dataset information for visual inspection.

    :param dataset: dict containing dataset information
    """
    print('\n\tData set information:\n\t{')
    for k,v in dataset.items():
        if k == 'heatmap_tm':
            print('\t\t{:<13} : {},'.format(k, 'N/A' if v is None else f'{np.max(v, axis=0)[0] + 1} measure point transitions'))
        elif hasattr(v, '__len__') and not isinstance(v, str):
            print('\t\t{:<13} : {},{}'.format(k, v, f' len({len(v)}),'))
        else:
            print(f'\t\t{k:<13} : {v},')
    print('\t}\n')



def add_s2sphere(lat, lng):
    """
    Add Cell ID by s2sphere

    :param lat: Latitude
    :param lng: Longitude
    :returns:   Unique cell ID 
    """

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

def reverse_s2sphere(cell_ids):
    """
    Convert s2sphehere Cell Ids to X and Y coordinate pairs

    :param cell_ids:    List of cell IDs
    :returns:           List of X and Y coordinates as tuples
    """
    #make list of cellIDs
    cellId = []
    for i in range(0, len(cell_ids)):
        cellId.append(cell_ids[i][0])
    cellId = list(map(int, cellId))

    #get lat and lng from CellIDs
    map_lat = []
    map_lng = []
    for i in range(0, len(cell_ids)):
        ll = str(s2sphere.CellId(cellId[i]).to_lat_lng())
        latlng = ll.split(',', 1)
        lat = latlng[0].split(':', 1)
        map_lat.append(float(lat[1]))
        map_lng.append(float(latlng[1]))
    
    pred_coordinates = list(zip(map_lat, map_lng))

    return pred_coordinates

def predict_synthetic_new(model, startpoints, lookback):
    """
    Given the trained model, predict trajectories of length 500 for the startpoints of the test set given a lookback length
    :param model:           Trained LSTM model
    :param startpoints:     Array of n startpoints of the test trajectories
    :param lookback:        Int of lookback n
    :returns:               Array of trajectories of length 500
    """

    while startpoints.shape[1] < 500:
        predict = model.predict(startpoints[:, -lookback:, :])
        predict = np.reshape(predict, (predict.shape[0], predict.shape[1], 1))
        startpoints = np.hstack((startpoints, predict))
    return startpoints



def shuffle_trips(df):
    """
    Shuffle trips in the dataset so trips of the same user end up in the train and in the test set
    :param df:      DataFrame sorted by user and trip ID
    :returns:       DataFrame with sorted trips but shuffled users
    """

    df = df.sort_values(by=['TRIP_ID|integer','sequence']) # sort trips by Trip ID and sequence number
    groups = [df for _, df in df.groupby('TRIP_ID|integer')] # group trips by trip ID
    random.shuffle(groups) # shuffle trips
    df = pd.concat(groups).reset_index(drop=True)
    return df

def train_test_split(df, train_size):
    """
    Split the DataFrame into a train and test DataFrame with a given train_size

    :param df:     Original DataFrame containing the whole dataset
    :returns:      Train and test DataFrame 
    """

    unique_trips = df['TRIP_ID|integer'].unique()
    train_trips, test_trips = np.split(unique_trips, [int(len(unique_trips)*train_size)])
    train = df.loc[df['TRIP_ID|integer'].isin(train_trips)]
    test = df.loc[df['TRIP_ID|integer'].isin(test_trips)]
    return train, test

def prepare_dataset(dataset_choice, batchsize, look_back, tesselation, epochs, trainsize, normalisation):
    """
    Provides data generators, labels and other information for selected dataset.

    :param dataset_choice:          which dataset to prepare
    :param targets:                 classification target
    :param batch_size:              batch size
    :param normalisation:           which normalisation method to use
    :param train_shuffle_repeat:    whether to shuffle and repeat the train generator
    :param categorical_labels:      whether to transform labels to categorical
    :param mp_heatmap:              whether to include a transition matrix for 64-Shot heatmap analyses
    :returns:                       dict containing train/eval/test generators, train/test labels, number of unique
                                    classes, original and transformed class ids, train/test steps, balanced class
                                    weights and data description
    :raises ValueError:             if dataset_choice is invalid
    """


    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        try:
            if dataset_choice == 0:
                data_path = dataset_configs['tapas_big_path']
                data_str = dataset_configs['tapas_big_str']
                data_name = dataset_configs['tapas_big_name']
            elif dataset_choice == 1:
                data_path = dataset_configs['tapas_small_path']
                data_str = dataset_configs['tapas_small_str']
                data_name = dataset_configs['tapas_small_name']
            else:
                raise ValueError('Invalid dataset parameter passed to prepare_dataset.')
        except KeyError as e:
            print(f'Missing dataset config key: {e}')
            sys.exit(1)

    if normalisation == 0:
        norm_function = no_normalisation
    elif normalisation == 1:
        norm_function = normalise_minmax

    if tesselation == 0:
        tess_function = 's2sphere'
    elif tesselation == 1:
        tess_function = 'h3'

    df = pd.read_csv(data_path)

    df = shuffle_trips(df)

    df_train, df_test = train_test_split(df, train_size=trainsize)

    train = interpolate_tesselation(df_train, tesselation= tess_function)
    test = interpolate_tesselation(df_test, tesselation= tess_function)
    startpoints = get_startpoints(test, look_back)

    #normalize and reshape
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_shape = train.shape
    train = scaler.fit_transform(train.reshape(-1,1))
    train = np.reshape(train, (train_shape))

    startpoints_shape = startpoints.shape
    startpoints = scaler.transform(startpoints.reshape(-1,1))
    startpoints = np.reshape(startpoints, (startpoints_shape))
    
    test_shape = test.shape
    test = scaler.transform(test.reshape(-1,1))
    test = np.reshape(test, (test_shape[0], test_shape[1], 1))

    trainX, trainY = lookback(train, previous=look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    train = np.reshape(train, (train_shape[0], train_shape[1], 1))

    return {
        'dataset_name' : data_name,
        'train_X'      : trainX,
        'train_Y'      : trainY,
        'test'         : test,
        'train'        : train,
        'startpoints'  : startpoints,
        'tesselation'  : tess_function,
        'lookback'     : look_back,
        'batch_size'   : batchsize,
        'norm_function': norm_function.__name__,
        'scaler'       : scaler,
        'epochs'       : epochs,
        'data_str'     : data_str}



def reverse_data(data, scaler, tesselation):
    """
    Inverse the Minmax scaler and inverse the tesselation back to XY coordinates
    :param data:    Normalized tesselated data
    :param scaler:  Fitted Min max scaler
    :tesselation:   Which tesselation was used
    :returns:       Array of XY coordinates
    """
    new_data = []
    for sample in data:
        sample = scaler.inverse_transform(sample)
        if tesselation == 's2sphere':
            sample = reverse_s2sphere(sample)
        elif tesselation == 'h3':
            sample = reverse_h3(sample)
        new_data.append(sample)

    return np.array(new_data)


def interpolate_tesselation(df, tesselation):
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
        if tesselation == 'h3':
            tesselation_cells = add_h3(X,Y)
        else:
            tesselation_cells = add_s2sphere(X,Y)
        data.append(tesselation_cells)
    return np.array(data)


def get_startpoints(data, lookback):
    """
    Given the test data, get the n startpoints of each test trajectory depending on the number of lookbacks used for the training data
    :param data:        Array containing the test data after normalisation and tesselation
    :param lookback:    Lookback number n 
    :returns:           Array containing n startpoints for each test trajectory
    """
    startpoints = []
    for trip in data:
        startpoints.append(trip[0:lookback])
    startpoints = np.array(startpoints)
    startpoints = np.reshape(startpoints, (startpoints.shape[0], startpoints.shape[1], 1))
    return startpoints



def lookback(df, previous=3):
    """
    Create data with a lookback

    :param df:          DataFrame with location data
    :param previous:    Lookback number
    :returns:           Tuple of arrays of X and Y data with a lookback  
    """
    dataX, dataY = [], []
    for traj in df:
        traj = np.array(traj)
        traj = traj.reshape(-1,1)     
        for i in range(len(traj)-previous):
            a = traj[i:(i+previous), 0]
            b = traj[i + previous, 0]
            dataX.append(a)
            dataY.append(np.array(b))
    return np.array(dataX), np.array(dataY)

def interpolate(x,y, num_points=500):
    """
    Interpolate a trajectory/sequence of X and Y coordinates to a fixed length of num_points.
    https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points/51515357

    :param X:           X coordinate
    :param Y:           Y coordinate
    :param num_points:  Length of the new trajectory/Number of X and Y coordinate pairs
    :returns:           Two lists of the length num_points with new X and Y coordinates
    """
    
    # Linear length on the line
    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    distance = distance/distance[-1]

    fx, fy = interp1d(distance, x), interp1d(distance, y)

    alpha = np.linspace(0, 1, num_points)
    x_regular, y_regular = fx(alpha), fy(alpha)

    return x_regular, y_regular



def save_csv(test, synth, train, train_=False):
    """
    Saves train, test and synthetic data as csv file

    :param test:    Test set
    :param train:   Train set
    :param synth:   Synthetic generated data
    :param train_:  If true, train data included
    :returns:       Saves csv file
    """

    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        predictions_path = dataset_configs['synthetic_csv_path']

    type = []
    coordinates = []

    m,n,r = test.shape
    for trip in range(0, m-1):
        type.append('original')
        coords = test[trip, :, :]
        coords = [tuple(x) for x in coords]
        coordinates.append(list(coords))

    m,n,r = synth.shape
    for trip in range(0,m-1):
        type.append('synthetic')
        coords = synth[trip, :, :]
        coords = [tuple(x) for x in coords]
        coordinates.append(list(coords))

    if train_==True:
        m,n,r = train.shape
        for trip in range(0,m-1):
            type.append('training data')
            coords = train[trip, :, :]
            coords = [tuple(x) for x in coords]
            coordinates.append(list(coords))

    data = pd.DataFrame(columns=['Type', 'Coordinates'])
    data['Type'] = type
    data['Coordinates'] = coordinates

    data.to_csv(predictions_path)


def save_as_geojson(test, synth, train, train_=False):
    """
    Saves train, test and synthetic data as geojson for visualisation in R

    :param test:    Test set
    :param train:   Train set
    :param synth:   Synthetic generated data
    :param train_:  If true, train data included
    :returns:       Saves geojson file
    """

    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        geojson_path = dataset_configs['geojson_path']

    type = []
    coordinates = []

    m,n,r = test.shape
    for trip in range(0, m-1):
        type.append('original')
        coords = test[trip, :, :]
        coords = [tuple(x) for x in coords]
        coordinates.append(LineString(coords))

    m,n,r = synth.shape
    for trip in range(0,m-1):
        type.append('synthetic')
        coords = synth[trip, :, :]
        coords = [tuple(x) for x in coords]
        coordinates.append(LineString(coords))

    if train_==True:
        m,n,r = train.shape
        for trip in range(0,m-1):
            type.append('training data')
            coords = train[trip, :, :]
            coords = [tuple(x) for x in coords]
            coordinates.append(LineString(coords))

    features = []

    for i in range(len(type)):
        features.append(Feature(geometry=coordinates[i], properties={"Type": type[i]}))

    feature_collection = FeatureCollection(features)
    with open(geojson_path, 'w') as f:
        dump(feature_collection, f)


def normalise_minmax(sample):
    """
    Normalises single sample according to minmax. Also strips wavelength information from sample.
    Adapted from Federico Malerba.

    :param sample:  sample to process
    :returns:       normalised sample
    """
    #normalize and reshape
    scaler = MinMaxScaler(feature_range=(0, 1))

    if np.max(sample) > 0:
        sample = sample.reshape(-1,1)
        sample = scaler.fit_transform(sample)
        return sample, scaler
    else:
        print('Sample is empty')


def no_normalisation(sample):
    """
    No normalisation

    :param sample:  sample to process
    :returns:       not normalised sample
    """
    return sample.reshape(-1,1), None


def trajectory_length(line):
    """
    Trajectory length of a LineString

    :param line:    Trajectory
    :returns:       Distance of trajectory in meters
    """
    return calculate_distance(line, Unit.meters)

