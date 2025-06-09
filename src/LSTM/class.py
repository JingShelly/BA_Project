import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import ETFs as etfs


data = pd.read_csv('LSTM/assets_wide.csv', parse_dates=['Date'])
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# data = data.dropna()
# display(data)


target = (data > data.shift(1)).astype(int)
def create_X_y_rolling(feature, target, window_size):
    X, y = [], []
    for i in range(len(feature) - window_size):
        window = feature.iloc[i:i+window_size]
        X.append(window)
    for i in range(len(target) - window_size):
         y.append(target.iloc[i + window_size])
    X = np.array(X)
    y = np.array(y)
    return X, y

window_size = 30
X, y = create_X_y_rolling(data, target, window_size)


dataset_size = len(X)
print(dataset_size)
train_size = int(dataset_size * 0.8)
test_size = int(dataset_size * 0.1)
val_size = dataset_size - train_size - test_size


X_train, X_val, X_test = X[:train_size], X[train_size: train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test= y[:train_size], y[train_size: train_size + val_size], y[train_size + val_size:]
X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=False),
    Dense(y_train.shape[1], activation='sigmoid')
])

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
register_keras_serializable()


model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['binary',]
)
cp = ModelCheckpoint(filepath='model_binary/best_binary.keras', 
                     save_best_only=True, monitor='val_loss', 
                     mode='min')
es = EarlyStopping(
    monitor='val_loss',        
    patience=10,                
    restore_best_weights=True 
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[cp, es]
)
# model.summary()


plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (crossentropy)')
plt.legend()
xticks = np.arange(1, 100, 5)
plt.xticks(xticks)
plt.title('Training and Validation Loss')
plt.show()


model = load_model('model_binary/best_binary.keras')
index_series = data.index[train_size + val_size + window_size:]

y_pred = model.predict(X_test)
# y_pred_mean = y_pred.mean(axis=0)
y_pred_classes = (y_pred > 0.5).astype(int)

Predict_result_df = pd.DataFrame(y_pred_classes, index=index_series, columns=data.columns)

true_test = target.iloc[train_size + val_size + window_size:]
ticker_name = 'SPY'

plt.figure(figsize=(10,6))
# plt.plot(Predict_result_df[ticker_name], label='Actual')
# plt.plot(true_test[ticker_name], label='Actual', color='blue')
plt.plot(Predict_result_df[ticker_name], label='Predicted', color='orange')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'LSTM regression model for {ticker_name}')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score

actual_flat = true_test.to_numpy().flatten()
predicted_flat = Predict_result_df.to_numpy().flatten()

accuracy = accuracy_score(actual_flat, predicted_flat)

print(f"Accuracy: {accuracy:.4f}")
