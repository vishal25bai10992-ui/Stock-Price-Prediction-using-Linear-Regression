# -----------------------------------------
# STEP 1: Import libraries
# -----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression


# -----------------------------------------
# STEP 2: Load stock data
# -----------------------------------------
stock = "AAPL"   # You can change this (e.g., TSLA, INFY.NS)

data = yf.download(stock, start="2020-01-01", end="2024-01-01")

# Use only closing price
data = data[['Close']]


# -----------------------------------------
# STEP 3: Prepare data
# -----------------------------------------
# Create a new column: next day price
data['Next_Day'] = data['Close'].shift(-1)

# Remove last row (it has NaN)
data = data.dropna()

# Input (X) = today's price
X = np.array(data[['Close']])

# Output (y) = next day's price
y = np.array(data['Next_Day'])


# -----------------------------------------
# STEP 4: Train model
# -----------------------------------------
model = LinearRegression()
model.fit(X, y)


# -----------------------------------------
# STEP 5: Predict next price
# -----------------------------------------
last_price = data[['Close']].iloc[-1].values.reshape(1, -1)

predicted_price = model.predict(last_price)

print("Last Price:", last_price[0][0])
print("Predicted Next Day Price:", predicted_price[0])


# -----------------------------------------
# STEP 6: Plot graph
# -----------------------------------------
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label="Actual Price")

# Predict all values for graph
predictions = model.predict(X)

plt.plot(predictions, label="Predicted Price")

plt.title("Simple Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()
