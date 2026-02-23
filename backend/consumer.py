#!/usr/bin/env python3

import json
import time
import logging
import os
import sys
import numpy as np
from datetime import datetime
from kafka import KafkaConsumer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from backend.config import KafkaConfig, MLConfig
from backend.ml.train_model import AnomalyDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataConsumer:
    def __init__(self, bootstrap_servers=None):
        servers = bootstrap_servers or KafkaConfig.BOOTSTRAP_SERVERS
        
        self.consumer = KafkaConsumer(
            KafkaConfig.RAW_DATA_TOPIC,
            bootstrap_servers=servers,
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            consumer_timeout_ms=5000,
            group_id=None
        )
        
        try:
            self.ml_model = AnomalyDetector.load(MLConfig.MODEL_PATH)
            logger.info("ML model loaded successfully")
        except FileNotFoundError:
            logger.warning("ML model not found. Predictions will be skipped.")
            self.ml_model = None
        
        self.messages_processed = 0
        self.anomalies_detected = 0
        logger.info(f"Consumer initialized. Topic: {KafkaConfig.RAW_DATA_TOPIC}")
    
    def preprocess_message(self, message):
        features = [
            message.get('temperature', 0),
            message.get('vibration', 0),
            message.get('pressure', 0),
            message.get('humidity', 0),
            message.get('power_consumption', 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def predict_anomaly(self, features):
        if self.ml_model is None:
            if features[0][0] > 85 or features[0][1] > 5:
                return 1, 0.9
            return 0, 0.1
        
        prediction = self.ml_model.predict(features)[0]
        confidence = self.ml_model.predict_proba(features)[0][prediction]
        return prediction, confidence
    
    def process_message(self, message):
        self.messages_processed += 1
        features = self.preprocess_message(message)
        prediction, confidence = self.predict_anomaly(features)
        
        if prediction == 1:
            self.anomalies_detected += 1
            logger.warning(
                f"ANOMALY DETECTED! ID: {message.get('id')}, "
                f"Equipment: {message.get('equipment_id')}, "
                f"Confidence: {confidence:.2f}"
            )
        
        if self.messages_processed % 100 == 0:
            logger.info(
                f"Processed: {self.messages_processed}, "
                f"Anomalies: {self.anomalies_detected}, "
                f"Rate: {self.anomalies_detected/self.messages_processed*100:.2f}%"
            )
        
        return {'original': message, 'prediction': int(prediction), 'confidence': float(confidence)}
    
    def consume(self, max_messages=None):
        logger.info("Starting consumer loop...")
        try:
            for message in self.consumer:
                data = message.value
                self.process_message(data)
                if max_messages and self.messages_processed >= max_messages:
                    break
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        finally:
            self.close()
    
    def close(self):
        logger.info(f"Closing consumer. Stats: {self.messages_processed} processed, {self.anomalies_detected} anomalies")
        self.consumer.close()


def main():
    consumer = DataConsumer()
    try:
        consumer.consume(max_messages=1000)
    except Exception as e:
        logger.error(f"Error in consumer: {e}")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()