import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("📈 Stock Price Prediction App")

# =============================
# Upload CSV
# =============================

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    # remove spaces in column names
    data.columns = data.columns.str.strip()

    st.subheader("Dataset Preview")
    st.write(data.head())

    # =============================
    # Select price column
    # =============================

    price_column = st.selectbox(
        "Select price column",
        data.columns
    )

    data = data[[price_column]]

    data.rename(columns={price_column: "Close"}, inplace=True)

    # remove commas and convert to float
    data['Close'] = data['Close'].astype(str).str.replace(",", "").astype(float)

    data.dropna(inplace=True)

    # =============================
    # Algorithm Selection
    # =============================

    algo = st.selectbox(
        "Select Algorithm",
        ["LSTM", "Decision Tree", "Linear Regression"]
    )

    # =============================
    # Train Button
    # =============================

    if st.button("Train Model"):

        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(data)

        x_train = []
        y_train = []

        for i in range(60, len(scaled_data)):
            x_train.append(scaled_data[i-60:i])
            y_train.append(scaled_data[i])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # =============================
        # LSTM
        # =============================

        if algo == "LSTM":

            model = Sequential()

            model.add(LSTM(50, return_sequences=True,
                           input_shape=(x_train.shape[1],1)))

            model.add(LSTM(50))

            model.add(Dense(1))

            model.compile(optimizer="adam", loss="mse")

            model.fit(x_train, y_train, epochs=5, batch_size=32)

            last_60 = scaled_data[-60:]

            last_60 = np.reshape(last_60, (1,60,1))

            pred = model.predict(last_60)

            predicted_price = scaler.inverse_transform(pred)[0][0]

        # =============================
        # Decision Tree
        # =============================

        elif algo == "Decision Tree":

            data['Lag1'] = data['Close'].shift(1)

            data.dropna(inplace=True)

            X = data[['Lag1']]
            y = data['Close']

            model = DecisionTreeRegressor()

            model.fit(X,y)

            predicted_price = model.predict(X.tail(1))[0]

        # =============================
        # Linear Regression
        # =============================

        else:

            data['Lag1'] = data['Close'].shift(1)

            data.dropna(inplace=True)

            X = data[['Lag1']]
            y = data['Close']

            model = LinearRegression()

            model.fit(X,y)

            predicted_price = model.predict(X.tail(1))[0]

        st.success(f"Predicted Price: {predicted_price}")

        # =============================
        # Graph
        # =============================

        fig, ax = plt.subplots()

        ax.plot(data['Close'], label="Actual Price")

        ax.scatter(len(data), predicted_price,
                   color="red",
                   label="Predicted Price")

        ax.legend()

        ax.set_title("Actual vs Predicted")

        st.pyplot(fig)