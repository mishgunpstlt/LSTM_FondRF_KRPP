import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Функция для расчета RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Функция для расчета MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

st.title("Прогнозирование цен акций с использованием LSTM")

# Список тикеров
tickers_list = ["GAZP", "SBER", "LKOH", "TATN", "MGNT", "YNDX", "NVTK", "VTBR", "ALRS", "POLY"]
ticker = st.selectbox("Выберите тикер актива:", tickers_list)

# Параметры обучения
st.sidebar.header("Параметры обучения")
epochs = st.sidebar.slider("Количество эпох:", min_value=10, max_value=500, value=100, step=10)
batch_size = st.sidebar.slider("Размер батча:", min_value=16, max_value=128, value=16, step=16)

# Отображение исторических данных
st.subheader("Исторические данные акций")

if ticker:
    api_url = f"http://127.0.0.1:5000/history?ticker={ticker}"
    response = requests.get(api_url)

    if response.status_code == 200:
        try:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df['Date'] = pd.to_datetime(df['Date'])
                st.write(df)

                # Добавление индикаторов
                # EMA
                df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
                df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()

                # Bollinger Bands
                window = 20
                df['SMA_20'] = df['Close'].rolling(window=window).mean()
                df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=window).std() * 2)
                df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=window).std() * 2)

                # RSI
                df['RSI'] = calculate_rsi(df['Close'], window=14)

                # MACD
                df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])

                # Или заполнение NaN значений для каждого столбца индивидуально
                df['SMA_20'].fillna(df['SMA_20'].mean(), inplace=True)
                df['BB_Upper'].fillna(df['BB_Upper'].mean(), inplace=True)
                df['BB_Lower'].fillna(df['BB_Lower'].mean(), inplace=True)
                df['RSI'].fillna(df['RSI'].mean(), inplace=True)
                df['MACD'].fillna(df['MACD'].mean(), inplace=True)
                df['MACD_Signal'].fillna(df['MACD_Signal'].mean(), inplace=True)
                df['MACD_Hist'].fillna(df['MACD_Hist'].mean(), inplace=True)

                # Построение графика с индикаторами
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Исторические данные'))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_10'], mode='lines', name='EMA 10', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA_30'], mode='lines', name='EMA 30', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
                st.plotly_chart(fig)

                # Прогнозирование с использованием LSTM
                st.subheader("Прогнозирование цен акций с использованием LSTM")

                # Нормализация данных
                scaler = MinMaxScaler(feature_range=(0, 1))
                df_scaled = scaler.fit_transform(df[['Close']])

                # Подготовка данных для LSTM
                def create_sequences(data, seq_length):
                    x = []
                    y = []
                    for i in range(len(data) - seq_length):
                        x.append(data[i:i + seq_length])
                        y.append(data[i + seq_length][0])  # Предсказываем только Close
                    return np.array(x), np.array(y)

                seq_length = 60
                x, y = create_sequences(df_scaled, seq_length)

                # Разделение на обучающую и тестовую выборки
                train_size = int(len(x) * 0.8)
                x_train, x_test = x[:train_size], x[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

                # Создание модели LSTM
                model = Sequential()
                model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                model.add(LSTM(100, return_sequences=False))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Обучение модели
                history = model.fit(
                    x_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_data=(x_test, y_test), 
                    verbose=1
                )

                # Прогнозирование
                predictions = model.predict(x_test)
                predictions_rescaled = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], x_test.shape[2] - 1))), axis=1))[:, 0]
                y_test_rescaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], x_test.shape[2] - 1))), axis=1))[:, 0]

                # Метрики
                mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
                mape = np.mean(np.abs((y_test_rescaled - predictions_rescaled) / y_test_rescaled)) * 100
                r2 = r2_score(y_test_rescaled, predictions_rescaled)

                # Вывод метрик
                st.subheader("Оценка качества модели")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MAPE: {mape:.2f}%")
                st.write(f"R²: {r2:.2f}")

                # Прогнозирование на следующие 30 дней
                forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]

                last_sequence = df_scaled[-seq_length:]
                last_sequence = np.expand_dims(last_sequence, axis=0)

                forecast = []
                for _ in range(30):
                    pred = model.predict(last_sequence)
                    forecast.append(pred[0, 0])
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1, 0] = pred

                forecast_rescaled = scaler.inverse_transform(
                    np.concatenate((np.array(forecast).reshape(-1, 1), np.zeros((len(forecast), x_test.shape[2] - 1))), axis=1)
                )[:, 0]

                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecast': forecast_rescaled
                })

                # Устранение разрыва между историческими данными и прогнозом
                last_known_date = df['Date'].iloc[-1]
                last_known_price = df['Close'].iloc[-1]

                forecast_with_last_known = pd.DataFrame({
                    'Date': [last_known_date] + list(forecast_df['Date']),
                    'Forecast': [last_known_price] + list(forecast_df['Forecast'])
                })

                # График прогноза
                fig_forecast = go.Figure()
                fig_forecast.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Исторические данные'))
                fig_forecast.add_trace(go.Scatter(x=forecast_with_last_known['Date'], y=forecast_with_last_known['Forecast'],
                                                  mode='lines', name='Прогноз', line=dict(color='red')))
                st.plotly_chart(fig_forecast)

            else:
                st.error("Получены некорректные данные.")
        except Exception as e:
            st.error(f"Ошибка обработки данных: {e}")
    else:
        st.error(f"Ошибка API: {response.status_code}")