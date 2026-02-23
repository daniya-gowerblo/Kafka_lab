"""
Конфигурация проекта Kafka Lab
"""

class KafkaConfig:
    """Настройки Kafka"""
    BOOTSTRAP_SERVERS = ['127.0.0.1:9092']
    RAW_DATA_TOPIC = 'industrial_raw_data'
    PROCESSED_DATA_TOPIC = 'industrial_processed_data'
    PREDICTIONS_TOPIC = 'industrial_predictions'
    CONSUMER_GROUP_ID = 'kafka_lab_group'
    AUTO_OFFSET_RESET = 'earliest'
    ACKS = 'all'
    RETRIES = 3

class DataConfig:
    """Настройки данных"""
    NUM_SAMPLES = 450000
    ANOMALY_RATIO = 0.05
    OUTPUT_PATH = 'data/raw_data.csv'
    FEATURES = [
        'temperature',
        'vibration',
        'pressure',
        'humidity',
        'power_consumption'
    ]

class MLConfig:
    """Настройки ML модели"""
    MODEL_PATH = 'backend/ml/trained_model.pkl'
    SCALER_PATH = 'backend/ml/scaler.pkl'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42