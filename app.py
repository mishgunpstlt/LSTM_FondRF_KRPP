import requests
import pandas as pd
import datetime
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

app = Flask(__name__)

# Глобальные переменные
scaler = MinMaxScaler()
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Получение данных с MOEX API
def fetch_moex_data(ticker, interval="24"):
    try:
        current_date = datetime.datetime.today()
        start_date = (current_date - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = current_date.strftime('%Y-%m-%d')

        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
        params = {"from": start_date, "till": end_date, "interval": interval}
        print(f"Запрос к MOEX: {url} с параметрами {params}")
        
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        print("Ответ MOEX:", data)

        # Проверяем, что ключи candles и data существуют
        candles = data.get('candles', {}).get('data', [])
        if not candles:
            raise ValueError(f"Нет данных для тикера {ticker}.")

        # Проверяем фактические столбцы
        columns = data['candles']['columns']  # Загружаем имена столбцов
        print("Доступные столбцы:", columns)

        # Преобразуем данные в DataFrame с актуальными столбцами
        df = pd.DataFrame(candles, columns=columns)
        print("Пример данных DataFrame:", df.head())

        # Преобразуем столбцы в нужный формат
        df['Close'] = df['close'].astype(float)  # Используем точное имя из columns
        df['Date'] = pd.to_datetime(df['begin'])  # Точное имя даты из columns
        df.set_index('Date', inplace=True)
        return df[['Close']]
    except requests.exceptions.RequestException as e:
        print(f"Сетевая ошибка: {e}")
        raise ValueError("Ошибка подключения к MOEX API.")
    except KeyError as e:
        print(f"Ошибка структуры ответа MOEX: {e}")
        raise ValueError("Ошибка структуры данных MOEX API.")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        raise ValueError("Ошибка при обработке данных MOEX.")

@app.route('/history', methods=['GET'])
def get_history():
    try:
        ticker = request.args.get('ticker')
        if not ticker:
            return jsonify({'error': 'Не указан тикер'})

        historical_data = fetch_moex_data(ticker)

        # Преобразуем данные для ответа
        records = historical_data.reset_index().to_dict(orient='records')
        print("Данные для графика:", records[:5])  # Печатаем первые 5 записей
        return jsonify(records)
    except Exception as e:
        print(f"Ошибка в эндпоинте /history: {e}")
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
