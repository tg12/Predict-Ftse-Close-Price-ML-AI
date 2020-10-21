'''THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY,
WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

# Bitcoin Cash (BCH)   qpz32c4lg7x7lnk9jg6qg7s4uavdce89myax5v5nuk
# Ether (ETH) -        0x843d3DEC2A4705BD4f45F674F641cE2D0022c9FB
# Litecoin (LTC) -     Lfk5y4F7KZa9oRxpazETwjQnHszEPvqPvu
# Bitcoin (BTC) -      34L8qWiQyKr8k4TnHDacfjbaSqQASbBtTd

# contact :- github@jamessawyer.co.uk



# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
# NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
# DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY,
# WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Dont forget to tip your server!
# Bitcoin Cash (BCH)   qpz32c4lg7x7lnk9jg6qg7s4uavdce89myax5v5nuk
# Ether (ETH) -        0x843d3DEC2A4705BD4f45F674F641cE2D0022c9FB
# Litecoin (LTC) -     Lfk5y4F7KZa9oRxpazETwjQnHszEPvqPvu
# Bitcoin (BTC) -      34L8qWiQyKr8k4TnHDacfjbaSqQASbBtTd

#Custom work available, See the Github Page! 

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


def rnn_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    model = Sequential()
    model.add(
        LSTM(
            input_shape=(
                layers[1],
                layers[0]),
            output_dim=layers[1],
            return_sequences=True))
    model.add(Dropout(params['dropout_keep_prob']))
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(params['dropout_keep_prob']))
    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("tanh"))

    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def predict_next_timestamp(model, history):
    """Predict the next time stamp given a sequence of history data"""

    prediction = model.predict(history)
    prediction = np.reshape(prediction, (prediction.size,))
    return prediction


def load_timeseries(filename, params):
    """Load time series dataset"""

    series = pd.read_csv(
        filename,
        sep=',',
        header=0,
        index_col=0,
        squeeze=True)
    data = series.values
    print(data)

    adjusted_window = params['window_size'] + 1

    # Split data into windows
    raw = []
    for index in range(len(data) - adjusted_window):
        raw.append(data[index: index + adjusted_window])

    # Normalize data
    result = normalize_windows(raw)

    raw = np.array(raw)
    result = np.array(result)

    # Split the input dataset into train and test
    split_ratio = round(params['train_test_split'] * result.shape[0])
    train = result[:int(split_ratio), :]
    np.random.shuffle(train)

    # x_train and y_train, for training
    x_train = train[:, :-1]
    y_train = train[:, -1]

    # x_test and y_test, for testing
    x_test = result[int(split_ratio):, :-1]
    y_test = result[int(split_ratio):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    x_test_raw = raw[int(split_ratio):, :-1]
    y_test_raw = raw[int(split_ratio):, -1]

    # Last window, for next time stamp prediction
    last_raw = [data[-params['window_size']:]]
    last = normalize_windows(last_raw)
    last = np.array(last)
    last = np.reshape(last, (last.shape[0], last.shape[1], 1))

    return [
        x_train,
        y_train,
        x_test,
        y_test,
        x_test_raw,
        y_test_raw,
        last_raw,
        last]


def normalize_windows(window_data):
    """Normalize data"""

    normalized_data = []
    for window in window_data:
        normalized_window = [((float(p) / float(window[0])) - 1)
                             for p in window]
        normalized_data.append(normalized_window)
    return normalized_data


def train_predict():
    """Train and predict time series data"""

    # Load command line arguments
    train_file = sys.argv[1]
    parameter_file = {
        "epochs": 100,
        "batch_size": 2,
        "window_size": 6,
        "train_test_split": 0.8,
        "validation_split": 0.1,
        "dropout_keep_prob": 0.2,
        "hidden_unit": 100
    }

    # Load training parameters
    params = json.loads(json.dumps(parameter_file))

    # Load time series dataset, and split it into train and test
    x_train, y_train, x_test, y_test, x_test_raw, y_test_raw,\
        last_window_raw, last_window = load_timeseries(train_file, params)

    # Build RNN (LSTM) model
    lstm_layer = [1, params['window_size'], params['hidden_unit'], 1]
    model = rnn_lstm(lstm_layer, params)

    # Train RNN (LSTM) model with train set
    model.fit(
        x_train,
        y_train,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        validation_split=params['validation_split'])

    # Check the model against test set
    predicted = predict_next_timestamp(model, x_test)
    predicted_raw = []
    for i in range(len(x_test_raw)):
        predicted_raw.append((predicted[i] + 1) * x_test_raw[i][0])

    # Plot graph: predicted VS actual
    plt.subplot(111)
    plt.plot(predicted_raw, label='Actual')
    plt.plot(y_test_raw, label='Predicted')
    plt.legend()
    plt.show()

    # Predict next time stamp
    next_timestamp = predict_next_timestamp(model, last_window)
    next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
    print('The next time stamp forecasting is: {}'.format(next_timestamp_raw))


if __name__ == '__main__':
    # python3 train_predict.py ./data.csv
    train_predict()
