import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# -----------------------
# 1. Download Historical Data
# -----------------------
tickers = ['GOOG', 'AAPL', 'IBM', 'SPY']
data = yf.download(tickers, start='2015-01-01', end='2024-06-01')['Close']
data = data.fillna(method='ffill')

# -----------------------
# 2. Feature Engineering
# -----------------------
window_size = 30
X = []
y = []
asset_labels = []

for ticker in tickers:
    prices = data[ticker].values
    returns = np.diff(prices)
    for i in range(window_size, len(prices)-1):
        window = prices[i-window_size:i]
        # Standardize window
        window_scaled = (window - np.mean(window)) / np.std(window)
        X.append(window_scaled)
        # Predict direction: 1 if next day’s return > 0, else 0
        label = 1 if returns[i] > 0 else 0
        y.append(label)
        asset_labels.append(ticker)

X = np.array(X)
y = np.array(y)

# -----------------------
# 3. Prepare Data
# -----------------------
# One-hot encode labels
y = to_categorical(y, num_classes=2)

# Reshape X for LSTM: [samples, time steps, features]
X = X.reshape((X.shape[0], window_size, 1))

# Train-test split (keeping time order)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
asset_labels_train, asset_labels_test = asset_labels[:split_idx], asset_labels[split_idx:]

# -----------------------
# 4. Build LSTM Model
# -----------------------
model = Sequential()
model.add(LSTM(50, input_shape=(window_size, 1)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# -----------------------
# 5. Train Model
# -----------------------
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# -----------------------
# 6. Evaluate
# -----------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# -----------------------
# 7. Plot Training History
# -----------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -----------------------
# 8. Predict on Test Set (optional)
# -----------------------
preds = model.predict(X_test)
predicted_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Save predictions with tickers
results_df = pd.DataFrame({
    'Asset': asset_labels_test,
    'Predicted': predicted_labels,
    'Actual': true_labels
})

print(results_df.head())
