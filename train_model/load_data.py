import paths
import torch
import datetime
import pandas as pd
import geopandas as gp

from apollo import mechanics as ma
from torch.autograd import Variable

### Set global model parameters
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DAYS = 6
FEATURE_LIST = ['Rain'] + ['Rain-' + f'{d+1}' for d in range(DAYS)] \
            + ['Temperature'] \
            + ['Temperature-' + f'{d+1}' for d in range(DAYS)] \
            + ['Resultant Windspeed'] \
            + ['Resultant Windspeed-' + f'{d+1}' for d in range(DAYS)] \
            + ['Humidity'] + ['Humidity-' + f'{d+1}' for d in range(DAYS)] \
            + ['Rain_28_Mu','Rain_90_Mu','Rain_180_Mu', 'Temperature_28_Mu','Temperature_90_Mu','Temperature_180_Mu']
            #+ ['Soil Moisture ' + f'{i+1}' for i in range(4)]

def load_data(filename, verbose=True):

    rf = pd.read_csv(filename)
    rf['Date'] = pd.to_datetime(rf['Date'], format='%Y-%m-%d').dt.date
    len_before = len(rf)
    rf = rf.dropna(subset=['Flow'])

    if verbose is True:
        print(len_before - len(rf), 'values are missing.')
        print('Mean flow is ', rf['Flow'].mean(), '+-', rf['Flow'].std())
        print('Maximum flow is ', rf['Flow'].max())
    return rf


def preprocess_data(rf, features, years_training):

    ###Test/Train data split by years
    rftrain = rf[~pd.to_datetime(rf['Date']).dt.year.isin(years_training)]

    ### Normalise features using parameters cached from the training set
    norm_cache = {}
    for f in features:
        rftrain[f] = ma.normalise(rftrain, f, norm_cache, write_cache=True)
        rf[f] = ma.normalise(rf, f, norm_cache, write_cache=False)

    rftrain['Date'] = rftrain['Date'].apply(
        lambda x: datetime.datetime.combine(x, datetime.datetime.min.time()).timestamp())
    rf['Date'] = rf['Date'].apply(lambda x: datetime.datetime.combine(x, datetime.datetime.min.time()).timestamp())

    ### Convert dataframe subsets to arrays and then to PyTorch variables
    trnset = rftrain.to_numpy()
    fullset = rf.to_numpy()

    return trnset, fullset


def reshape_input(set, xspace):
    X = set[:, xspace].reshape(len(set), len(xspace)).astype(float)
    return X #torch.from_numpy(X).to(device)

def reshape_output(set, yspace):
    Y = set[:, yspace].reshape(len(set), 1).astype(float)
    return Y #torch.from_numpy(Y).to(device)



def process_station(station_nr, features=FEATURE_LIST):

    input_data_fp = paths.CATCHMENT_BASINS + '/' + str(station_nr) + '/' + str(str(station_nr) + '_lumped.csv')

    rf = load_data(input_data_fp)
    trnset, full_set = preprocess_data(rf, features, [2010 + i for i in range(12)])

    targets = ['Flow']
    xspace = ma.featurelocator(rf, features)
    yspace = ma.featurelocator(rf, targets)

    x_train, y_train = get_train_data(trnset, xspace, yspace)
    y_test = get_test_data(full_set, xspace)


    #print(boundary.geometry.area/1000)