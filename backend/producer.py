#!/usr/bin/env python3
"""
Kafka Producer для отправки данных промышленного оборудования в реальном времени
"""

import json
import time
import random
import logging
import os
import sys
from datetime import datetime

# === ВАЖНО: Добавляем корень проекта в путь поиска Python ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
# ============================================================

from kafka import KafkaProducer
from backend.config import KafkaConfig, DataConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProducer:
    """Producer класс для отправки данных в Kafka"""
    
    def __init__(self, bootstrap_servers=None):
        servers = bootstrap_servers or KafkaConfig.BOOTSTRAP_SERVERS
        
        self.producer = KafkaProducer(
            bootstrap_servers=servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8') if k else None,
            acks=KafkaConfig.ACKS,
            retries=KafkaConfig.RETRIES,
            request_timeout_ms=30000
        )
        
        self.topic = KafkaConfig.RAW_DATA_TOPIC
        self.messages_sent = 0
        logger.info(f"Producer initialized. Topic: {self.topic}")
    
    def generate_sample(self, equipment_id=None):
        """Генерация одного сэмпла данных"""
        
        if equipment_id is None:
            equipment_id = random.randint(1, 10)
        
        is_anomaly = random.random() < DataConfig.ANOMALY_RATIO
        
        if is_anomaly:
            sample = {
                'id': self.messages_sent + 1,
                'timestamp': datetime.now().isoformat(),
                'equipment_id': equipment_id,
                'temperature': round(random.uniform(85, 100), 2),
                'vibration': round(random.uniform(5, 8), 2),
                'pressure': round(random.choice([random.uniform(3, 6), random.uniform(15, 20)]), 2),
                'humidity': round(random.choice([random.uniform(10, 25), random.uniform(80, 95)]), 2),
                'power_consumption': round(random.uniform(180, 250), 2),
                'is_anomaly': True
            }
        else:
            sample = {
                'id': self.messages_sent + 1,
                'timestamp': datetime.now().isoformat(),
                'equipment_id': equipment_id,
                'temperature': round(random.gauss(70, 5), 2),
                'vibration': round(random.gauss(3, 0.8), 2),
                'pressure': round(random.gauss(10, 1.5), 2),
                'humidity': round(random.gauss(50, 8), 2),
                'power_consumption': round(random.gauss(125, 20), 2),
                'is_anomaly': False
            }
        
        return sample
    
    def send_message(self, message, key=None):
        """Отправка одного сообщения в Kafka"""
        try:
            future = self.producer.send(self.topic, key=key, value=message)
            future.get(timeout=10)
            self.messages_sent += 1
            if self.messages_sent % 100 == 0:
                logger.info(f"Sent {self.messages_sent} messages")
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def stream_live(self, duration_seconds=60, samples_per_second=20):
        """Генерация и отправка live-данных"""
        logger.info(f"Starting live stream for {duration_seconds}s at {samples_per_second} samples/sec")
        start_time = time.time()
        interval = 1.0 / samples_per_second
        
        while time.time() - start_time < duration_seconds:
            sample = self.generate_sample()
            self.send_message(sample, key=str(sample['equipment_id']))
            time.sleep(interval)
        
        logger.info(f"Live stream completed. Messages sent: {self.messages_sent}")
    
    def flush_and_close(self):
        logger.info("Flushing and closing producer...")
        self.producer.flush()
        self.producer.close()
        logger.info(f"Producer closed. Total messages: {self.messages_sent}")


def main():
    producer = DataProducer()
    try:
        producer.stream_live(duration_seconds=60, samples_per_second=20)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        producer.flush_and_close()


if __name__ == "__main__":
    main()