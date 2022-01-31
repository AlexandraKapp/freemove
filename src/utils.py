from geojson_length import calculate_distance, Unit
import numpy as np
import yaml
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import s2sphere
from geojson import LineString, Feature, FeatureCollection, dump
import itertools
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

def train_test_split(df, train_size=0.8):
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

def prepare_dataset(dataset_choice, batchsize, look_back, tesselation, epochs, normalisation=2):
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

    df = pd.read_csv(data_path)

    df = shuffle_trips(df)

    #train test split
    train, startpoints = split_data(df)
    
    #normalize and reshape
    train, train_scaler = norm_function(train)
    startpoints, start_scaler = norm_function(startpoints)

    test = save_test_set(df)

    #Create dataset with lookback
    #look_back=2
    trainX, trainY = data_lookback(train, look_back)

    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))


    # list of "class" names used for confusion matrices and validity testing. Not always classes, also subgroups or
    # minerals, depending on classification targets
    return {
        'dataset_name' : data_name,
        'train_X'      : trainX,
        'train_Y'      : trainY,
        'test'         : test,
        'startpoints'  : startpoints,
        'tesselation'  : tesselation,
        'batch_size'   : batchsize,
        'norm_function': norm_function.__name__,
        'train_scaler' : train_scaler,
        'start_scaler' : start_scaler,
        'epochs'       : epochs,
        'data_str'     : data_str}



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


def split_data(df, split=0.9):
    """
    Split dataset in train and test

    :param df:  DataFrame containing the whole dataset
    :split:     Split for train and test
    :returns:   Array of train dataset and array of startpoints of the test dataset
    """
    #TODO split users into diff datasets?
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
         

def data_lookback(df, previous=1):
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

def add_cellID(lat, lng):
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


def add_h3(lat,lng, resolution=13):
    """
    Add unique cell Id by Uber H3 for each coordinate pair
    
    :param lat:         List of Latitudes
    :param lng:         List of Longitudes
    :param resolution:  Controls the size of the hexagon and the number of unique indexes (0 is coarsest and 15 is finest resolution), see https://h3geo.org/docs/core-library/restable/
    :returns:           List with h3 indexes of the coordinates
    """
    h3_indexes = []

    for i in range(0, len(lat)):
        index = h3.geo_to_h3(lat[i], lng[i], resolution)
        h3_indexes.append(index)
    
    #return h3.geo_to_h3(lat, lng, resolution) 
    return h3_indexes

def save_test_set(df, split=0.8):
    """
    Returns a list of coordinates in tuple form
    """

    num_trips = len(df['TRIP_ID|integer'].unique())
    counter = 0
    test = []
    for tripID in df['TRIP_ID|integer'].unique():
            counter +=1
            trip = df.loc[df['TRIP_ID|integer'] == tripID]
            X, Y = interpolate(trip['X'],trip['Y'], num_points=500)
            if counter < num_trips*split:
                pass
            else:
                for pair in ((X[i], Y[i]) for i in range(min(len(X), len(Y)))):
                    test.append(pair)  

    return test

def generate_output_df(test_coordinates, pred_coordinates, num_points=500):
    
    #divide into trips and add type and append to dataframe
    coordinates = []
    type = []

    for x in range(1,int(len(test_coordinates)/num_points)+1):
        num = num_points*x
        coords = test_coordinates[:num]
        type.append('original')
        coordinates.append(LineString(coords))

    for x in range(1,int(len(pred_coordinates)/num_points)+1):
        num = num_points*x
        coords = pred_coordinates[:num]
        type.append('synthetic')
        coordinates.append(LineString(coords))

    data = pd.DataFrame(columns=['Type', 'Coordinates'])
    data['Type'] = type
    data['Coordinates'] = coordinates
    
    return data

def save_results(df):
    #output_df.to_csv('/Users/jh/github/freemove/data/synthetic_data.csv')
    #save synthetic trajectories to csv
    #pred_df = pd.DataFrame.from_dict(output)
    #pred_df.to_csv('/Users/jh/github/freemove/data/predictions_both.csv')


    with open('config/datasets.yaml') as cnf:
        dataset_configs = yaml.safe_load(cnf)
        predictions_path = dataset_configs['predictions_path']
        predictions_str = dataset_configs['predictions_str']
        predictions_name = dataset_configs['predictions_name']
        geo_path = dataset_configs['geo_path']
        geo_str = dataset_configs['geo_str']
        geo_name = dataset_configs['geo_name']

    save_as_geojson(df, geo_path)
    df.to_csv(predictions_path)


def save_as_geojson(df, path):

    features = []
    type = df['Type']
    coordinates = df['Coordinates']
    for i in range(len(df)):
        features.append(Feature(geometry=coordinates[i], properties={"Type": type[i]}))

    feature_collection = FeatureCollection(features)
    with open(path, 'w') as f:
        dump(feature_collection, f)


def predict_synthetic(model, startpoints, scaler, tesselation):
    
    trajectory = []

    #predict synthetic data
    for point in startpoints:
        point = np.array(point)
        point = point.reshape(-1,1)
        predict = point
        #predict = model.predict(point)

        #trajectory = []
        if scaler == None:
            trajectory.append(predict)
        else:
            trajectory.append(scaler.inverse_transform(predict)) #inverse normalization
        for i in range(0,499):
            p = np.reshape(predict, (predict.shape[0], predict.shape[1], 1))
            predict = model.predict(p)
            if scaler==None:
                trajectory.append(predict)
            else:
                trajectory.append(scaler.inverse_transform(predict)) #inverse normalization

    s_c_id = list(itertools.chain(*trajectory)) #change to one iterable

    if tesselation == 1:
        pass
    if tesselation == 0:
        pred_coordinates = reverse_s2sphere(s_c_id)

    return pred_coordinates

def reverse_h3(h3_indexes):
    """
    Convert h3 indexes to X and Y coordinate pairs
    
    :param h3_indexes:  List of h3 indexes
    :returns:           List of X and Y coordinates
    """
    coordinates = []
    for index in h3_indexes:
        geo = h3.h3_to_geo(s_c_id[index])
        coordinates.append(geo)
    return coordinates

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

