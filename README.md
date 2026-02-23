# Industrial Equipment Monitoring with Apache Kafka

Система мониторинга промышленного оборудования с обнаружением аномалий в реальном времени на основе Apache Kafka и ML.

## Описание проекта

Система реализует полный pipeline обработки данных промышленного оборудования:
- **Генерация данных** → Producer отправляет показания датчиков (температура, вибрация, давление, влажность, потребление энергии)
- **Передача данных** → Apache Kafka (KRaft mode) принимает и хранит сообщения
- **Обработка данных** → Consumer читает сообщения и применяет ML-модель для обнаружения аномалий
- **Визуализация** → Streamlit Dashboard показывает данные в реальном времени

## Архитектура

Producer (Python) -> Apache Kafka (KRaft, 1 broker) -> Consumer (Python + ML model) -> Dashboard (Streamlit)


## Требования

- Docker и Docker Compose
- Python 3.10+
- Git

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd kafka_lab

### 2. Установка зависимостей Python

pip install -r requirements.txt

### 3. Запуск Kafka (Docker)

# Запустить Kafka и Kafka UI
docker compose up -d

# Подождать 60-90 секунд для инициализации

# Проверить статус
docker compose ps

# Ожидаемый результат: 
kafka-1 в статусе Up (healthy)

### 4. Обучение ML-модели

py backend/ml/train_model.py

# Создаст файлы:
# backend/ml/trained_model.pkl - обученная модель Random Forest
# backend/ml/scaler.pkl - scaler для нормализация данных

### 5. Подготовка датасета

# Файл data/raw_data.csv не включен в репозиторий из-за размера. Если он отсутствует, сгенерируйте данные:

py data/generate_dataset.py

### 6. Генерация данных и отправка в Kafka

cd kafka_lab
py backend/producer.py

# Ожидаемый результат: Sent .. messages

py backend/consumer.py

# Ожидаемый результат: 
# ML model loaded succesfully
# ANOMALY DETECTED! 
# Processed: .., Anomalies: 44

### 7. Запуск Dashboard

py -m streamlit run frontend/app.py

# Откройте: http://localhost:8501

# Ожидаемый результат: интерактивные графики с данными

## Структура проекта

kafka_lab/
├── backend/
│   ├── config.py              # Конфигурация Kafka и ML
│   ├── producer.py            # Producer для отправки данных
│   ├── consumer.py            # Consumer с ML-моделью
│   └── ml/
│       ├── train_model.py     # Обучение модели
│       ├── trained_model.pkl  # Обученная модель
│       └── scaler.pkl         # Scaler для нормализации
├── frontend/
│   └── app.py                 # Streamlit Dashboard
├── data/
│   ├── raw_data.csv           # Готовый датасет (не включен в репозиторий)
│   └── generate_dataset.py    # Генератор
├── docker-compose.yml         # Docker конфигурация
├── requirements.txt           # Python зависимости
└── README.md                  # Этот файл

