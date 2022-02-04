import argparse
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from utils import predict_synthetic_new, print_dataset_info, repeat_and_collate, prepare_dataset, reverse_data, save_as_geojson, save_csv

def classify(**args):
    """
    Main method that prepares dataset, builds model, executes training and displays results.

    :param args: keyword arguments passed from cli parser
    """
    # only allow print-outs if execution has no repetitions
    allow_print = args['repetitions'] == 1
    # determine classification targets and parameters to construct datasets properly
    d = prepare_dataset(
        args['dataset_choice'],
        args['batch_size'],
        args['lookback_choice'],
        args['tess_choice'],
        args['epochs'],
        args['trainsize'],
        args['norm_choice'])

    print_dataset_info(d)

  # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(d['train_X'].shape[1], d['train_X'].shape[2]), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    if allow_print:
        model.summary()
        print('')

    # train and evaluate
    model.fit(d['train_X'], d['train_Y'], epochs=d['epochs'], batch_size=d['batch_size'], verbose=2)

    pred = predict_synthetic_new(model, d['startpoints'], d['lookback'])
    
    pred = reverse_data(pred, d['scaler'], d['tesselation'])
    test = reverse_data(d['test'], d['scaler'], d['tesselation'])
    train =reverse_data(d['train'], d['scaler'], d['tesselation'])


    save_as_geojson(test,pred,train,train_=True)
    save_csv(test,pred,train,train_=False)
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--dataset',
        type=int,
        choices=[0, 1],
        default=1,
        help='Which dataset to use. 0=big, 1=small',
        dest='dataset_choice'
    )

    parser.add_argument(
        '-r', '--repetitions',
        type=int,
        default=1,
        help='Number of times to repeat experiment',
        dest='repetitions'
    )

    parser.add_argument(
        '-b', '--batchsize',
        type=int,
        default=500,
        help='Target batch size of dataset preprocessing',
        dest='batch_size'
    )

    parser.add_argument(
        '-t', '--tesselation',
        type=int,
        choices=[0, 1],
        default=0,
        help='Which tesselation package. 0=s2sphere, 1=h3',
        dest='tess_choice'
    )

    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=10,
        help='How many epochs to train for',
        dest='epochs'
    )

    parser.add_argument(
        '-lb', '--lookback',
        type=int,
        default=10,
        help='Number of lookbacks in dataset',
        dest='lookback_choice'

    )

    parser.add_argument(
    '-s', '--trainsize',
    type=int,
    default=0.80,
    help='Size of the training data split',
    dest='trainsize'

    )

    parser.add_argument(
        '-n', '--normalisation',
        type=int,
        choices=[0, 1],
        default=1,
        help='Which normalisation to use. 0=None, 1=minmax',
        dest='norm_choice'
    )

    args = parser.parse_args()

    repeat_and_collate(classify, **vars(args))
