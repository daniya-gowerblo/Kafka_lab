#!/usr/bin/env python3
"""
Генератор датасета для промышленного оборудования
Создаёт 450 000+ сэмплов с 5 признаками для задачи обнаружения аномалий
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Добавляем backend в path для импорта config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import DataConfig

# Фиксируем random seed для воспроизводимости
np.random.seed(42)


def generate_industrial_data():
    """Генерация реалистичных данных промышленного оборудования"""
    
    print("Генерация датасета...")
    print(f"Количество сэмплов: {DataConfig.NUM_SAMPLES:,}")
    
    # Генерация временных меток (реальное время с интервалом 1 секунда)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(DataConfig.NUM_SAMPLES)]
    
    # Генерация ID оборудования (10 единиц оборудования)
    equipment_ids = np.random.randint(1, 11, DataConfig.NUM_SAMPLES)
    
    # === ГЕНЕРАЦИЯ ПРИЗНАКОВ (5 признаков) ===
    
    # 1. Температура (°C) — нормальное значение 65-75
    temperature = np.random.normal(loc=70, scale=5, size=DataConfig.NUM_SAMPLES)
    
    # 2. Вибрация (mm/s) — нормальное значение 2-4
    vibration = np.random.normal(loc=3, scale=0.8, size=DataConfig.NUM_SAMPLES)
    
    # 3. Давление (бар) — нормальное значение 8-12
    pressure = np.random.normal(loc=10, scale=1.5, size=DataConfig.NUM_SAMPLES)
    
    # 4. Влажность (%) — нормальное значение 40-60
    humidity = np.random.normal(loc=50, scale=8, size=DataConfig.NUM_SAMPLES)
    
    # 5. Потребление энергии (кВт) — нормальное значение 100-150
    power_consumption = np.random.normal(loc=125, scale=20, size=DataConfig.NUM_SAMPLES)
    
    # === ГЕНЕРАЦИЯ АНОМАЛИЙ ===
    
    # Создаём метку аномалии (0 = норма, 1 = аномалия)
    anomaly = np.zeros(DataConfig.NUM_SAMPLES, dtype=int)
    anomaly_indices = np.random.choice(
        DataConfig.NUM_SAMPLES, 
        size=int(DataConfig.NUM_SAMPLES * DataConfig.ANOMALY_RATIO), 
        replace=False
    )
    anomaly[anomaly_indices] = 1
    
    # Добавляем аномальные значения в данные
    # Температура аномально высокая (> 85°C)
    temperature[anomaly_indices] = np.random.uniform(85, 100, size=len(anomaly_indices))
    
    # Вибрация аномально высокая (> 5 mm/s)
    vibration[anomaly_indices] = np.random.uniform(5, 8, size=len(anomaly_indices))
    
    # Давление аномально низкое или высокое
    pressure[anomaly_indices] = np.where(
        np.random.rand(len(anomaly_indices)) > 0.5,
        np.random.uniform(3, 6, size=len(anomaly_indices)),
        np.random.uniform(15, 20, size=len(anomaly_indices))
    )
    
    # Влажность аномальная
    humidity[anomaly_indices] = np.where(
        np.random.rand(len(anomaly_indices)) > 0.5,
        np.random.uniform(10, 25, size=len(anomaly_indices)),
        np.random.uniform(80, 95, size=len(anomaly_indices))
    )
    
    # Потребление энергии аномальное
    power_consumption[anomaly_indices] = np.random.uniform(180, 250, size=len(anomaly_indices))
    
    # === СОЗДАНИЕ DATAFRAME ===
    
    df = pd.DataFrame({
        'id': range(1, DataConfig.NUM_SAMPLES + 1),
        'timestamp': timestamps,
        'equipment_id': equipment_ids,
        'temperature': np.round(temperature, 2),
        'vibration': np.round(vibration, 2),
        'pressure': np.round(pressure, 2),
        'humidity': np.round(humidity, 2),
        'power_consumption': np.round(power_consumption, 2),
        'anomaly': anomaly  # Целевая переменная
    })
    
    return df


def save_data(df, output_path=None):
    """Сохранение данных в CSV"""
    
    if output_path is None:
        output_path = DataConfig.OUTPUT_PATH
    
    # Создаем папку data если не существует
    os.makedirs('data', exist_ok=True)
    
    # Сохраняем
    df.to_csv(output_path, index=False)
    
    print(f"Данные сохранены в: {output_path}")
    print(f"Размер файла: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # Статистика
    print(f"\nСтатистика датасета:")
    print(f"   - Всего сэмплов: {len(df):,}")
    print(f"   - Нормальных: {(df['anomaly'] == 0).sum():,} ({(df['anomaly'] == 0).mean()*100:.1f}%)")
    print(f"   - Аномалий: {(df['anomaly'] == 1).sum():,} ({(df['anomaly'] == 1).mean()*100:.1f}%)")
    print(f"   - Признаков: {len(DataConfig.FEATURES)}")
    print(f"   - Период: {df['timestamp'].min()} до {df['timestamp'].max()}")


def main():
    """Основная функция"""
    print("=" * 60)
    print("ГЕНЕРАТОР ДАННЫХ ПРОМЫШЛЕННОГО ОБОРУДОВАНИЯ")
    print("=" * 60)
    
    # Генерация
    df = generate_industrial_data()
    
    # Сохранение
    save_data(df)
    
    # Предпросмотр
    print(f"\nПервые 5 строк:")
    print(df.head())
    
    print("\n" + "=" * 60)
    print("Генерация завершена успешно!")
    print("=" * 60)


if __name__ == "__main__":
    main()