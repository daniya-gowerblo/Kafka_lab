#!/usr/bin/env python3
"""
ML модель для обнаружения аномалий в данных промышленного оборудования
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
# ============================================================

from backend.config import MLConfig, DataConfig


class AnomalyDetector:
    """
    Класс для обучения и использования ML модели обнаружения аномалий
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Инициализация модели
        
        Args:
            model_type: 'isolation_forest' или 'random_forest'
        """
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = DataConfig.FEATURES
        self.is_fitted = False
        
    def load_data(self, csv_path):
        """Загрузка и подготовка данных"""
        
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        #извлекаем признаки и целевую переменную
        X = df[DataConfig.FEATURES].values
        y = df['anomaly'].values
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Anomaly distribution: {np.bincount(y)}")
        
        return X, y
    
    def preprocess(self, X, fit=False):
        """Препроцессинг: масштабирование признаков"""
        
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def train(self, X_train, y_train):
        """Обучение модели"""
        
        print(f"Training {self.model_type} model...")
        
        if self.model_type == 'isolation_forest':
            # Unsupervised подход — хорошо для аномалий
            self.model = IsolationForest(
                n_estimators=100,
                contamination=DataConfig.ANOMALY_RATIO,
                random_state=MLConfig.RANDOM_STATE,
                n_jobs=-1
            )
            # Isolation Forest использует -1 для аномалий
            y_train_if = np.where(y_train == 1, -1, 1)
            self.model.fit(X_train)
            
        elif self.model_type == 'random_forest':
            # Supervised подход — если есть размеченные данные
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=MLConfig.RANDOM_STATE,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        print("Model training completed")
    
    def evaluate(self, X_test, y_test):
        """Оценка качества модели"""
        
        if not self.is_fitted:
            print("Model not trained yet!")
            return None
        
        print("Evaluating model...")
        
        #предсказания
        if self.model_type == 'isolation_forest':
            #isolation Forest возвращает -1 для аномалий
            y_pred_raw = self.model.predict(X_test)
            y_pred = np.where(y_pred_raw == -1, 1, 0)
            #для ROC AUC нужен score
            y_scores = -self.model.score_samples(X_test)
        else:
            y_pred = self.model.predict(X_test)
            y_scores = self.model.predict_proba(X_test)[:, 1]
        
        #метрики
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        #ROC AUC
        try:
            roc_auc = roc_auc_score(y_test, y_scores)
            print(f"ROC-AUC Score: {roc_auc:.4f}")
        except:
            pass
        
        return {
            'predictions': y_pred,
            'scores': y_scores,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict(self, X):
        """Предсказание для новых данных"""
        
        if not self.is_fitted:
            raise RuntimeError("Model not trained!")
        
        X_scaled = self.preprocess(X)
        
        if self.model_type == 'isolation_forest':
            y_pred_raw = self.model.predict(X_scaled)
            return np.where(y_pred_raw == -1, 1, 0)
        else:
            return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Вероятности предсказания"""
        
        if not self.is_fitted:
            raise RuntimeError("Model not trained!")
        
        X_scaled = self.preprocess(X)
        
        if self.model_type == 'isolation_forest':
            scores = -self.model.score_samples(X_scaled)
            probs_anomaly = np.clip((scores - scores.min()) / (scores.max() - scores.min() + 1e-10), 0, 1)
            return np.column_stack([1 - probs_anomaly, probs_anomaly])
        else:
            return self.model.predict_proba(X_scaled)
    
    def save(self, model_path=None, scaler_path=None):
        """Сохранение модели и скалера"""
        
        if not self.is_fitted:
            print("Model not trained, nothing to save")
            return
        
        model_path = model_path or MLConfig.MODEL_PATH
        scaler_path = scaler_path or MLConfig.SCALER_PATH
        
        # Создаём папку если нет
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Сохраняем
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    @classmethod
    def load(cls, model_path=None, scaler_path=None):
        """Загрузка обученной модели"""
        
        model_path = model_path or MLConfig.MODEL_PATH
        scaler_path = scaler_path or MLConfig.SCALER_PATH
        
        detector = cls()
        detector.model = joblib.load(model_path)
        detector.scaler = joblib.load(scaler_path)
        detector.is_fitted = True
        
        print(f"Model loaded from {model_path}")
        return detector


def train_and_save():
    """Основная функция: обучение и сохранение модели"""
    
    print("TRAINING ANOMALY DETECTION MODEL")
    
    #загрузка данных
    detector = AnomalyDetector(model_type='random_forest')
    X, y = detector.load_data(DataConfig.OUTPUT_PATH)
    
    #препроцессинг
    X_scaled = detector.preprocess(X, fit=True)
    
    #разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=MLConfig.TEST_SIZE, 
        stratify=y,
        random_state=MLConfig.RANDOM_STATE
    )
    
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    #обучение
    detector.train(X_train, y_train)
    
    #оценка
    detector.evaluate(X_test, y_test)
    
    #сохранение
    detector.save()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train_and_save()