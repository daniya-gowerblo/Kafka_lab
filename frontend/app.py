#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from kafka import KafkaConsumer
from backend.config import KafkaConfig, DataConfig

st.set_page_config(
    page_title="Kafka Lab Dashboard",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Industrial Equipment Monitoring")
st.markdown("*Real-time anomaly detection via Apache Kafka*")

st.sidebar.header("Settings")

bootstrap_servers = st.sidebar.text_input(
    "Kafka Bootstrap Servers", 
    value="127.0.0.1:9092"
)

topic_options = [
    KafkaConfig.RAW_DATA_TOPIC,
    KafkaConfig.PROCESSED_DATA_TOPIC,
    KafkaConfig.PREDICTIONS_TOPIC
]
selected_topic = st.sidebar.selectbox("Select Topic", topic_options)

max_messages = st.sidebar.slider("Max messages to display", 100, 5000, 1000)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 60, 5)

def fetch_kafka_messages(topic, servers, max_msgs):
    messages = []
    
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[servers] if isinstance(servers, str) else servers,
            group_id=None,
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            consumer_timeout_ms=3000,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        for i, message in enumerate(consumer):
            if i >= max_msgs:
                break
            messages.append(message.value)
            
        consumer.close()
        
    except Exception as e:
        st.sidebar.error(f"Connection error: {e}")
        
    return messages

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Status", "Connected" if bootstrap_servers else "Disconnected")

with col2:
    st.metric("Topic", selected_topic)

with col3:
    st.metric("Messages", "Loading...")

with col4:
    st.metric("Anomalies", "Loading...")

if st.button("Refresh Data") or auto_refresh:
    with st.spinner("Fetching data from Kafka..."):
        messages = fetch_kafka_messages(selected_topic, bootstrap_servers, max_messages)
    
    if messages:
        df = pd.DataFrame(messages)
        
        col3.metric("Messages", len(df))
        anomaly_count = df.get('is_anomaly', df.get('anomaly', pd.Series([0]*len(df)))).sum() if 'is_anomaly' in df.columns or 'anomaly' in df.columns else 0
        col4.metric("Anomalies", int(anomaly_count))
        
        st.subheader("Temperature Over Time")
        if 'timestamp' in df.columns and 'temperature' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            fig_temp = px.line(
                df.tail(500), 
                x='timestamp', 
                y='temperature',
                color='is_anomaly' if 'is_anomaly' in df.columns else None,
                title="Equipment Temperature",
                color_discrete_map={True: 'red', False: 'blue', None: 'blue'}
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        st.subheader("Feature Distributions")
        feature_cols = DataConfig.FEATURES
        if all(col in df.columns for col in feature_cols):
            fig_dist = px.histogram(
                df, 
                x=feature_cols[0], 
                color='is_anomaly' if 'is_anomaly' in df.columns else None,
                marginal='box',
                title=f"{feature_cols[0].title()} Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        st.subheader("Feature Correlations")
        if all(col in df.columns for col in feature_cols[:3]):
            fig_scatter = px.scatter_matrix(
                df.tail(200),
                dimensions=feature_cols[:3],
                color='is_anomaly' if 'is_anomaly' in df.columns else None,
                title="Feature Relationships"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.subheader("Anomalies by Equipment")
        if 'equipment_id' in df.columns and ('is_anomaly' in df.columns or 'anomaly' in df.columns):
            anomaly_col = 'is_anomaly' if 'is_anomaly' in df.columns else 'anomaly'
            equip_anomalies = df.groupby('equipment_id')[anomaly_col].sum().reset_index()
            fig_equip = px.bar(
                equip_anomalies,
                x='equipment_id',
                y=anomaly_col,
                title="Anomaly Count per Equipment",
                labels={'equipment_id': 'Equipment ID', anomaly_col: 'Anomaly Count'}
            )
            st.plotly_chart(fig_equip, use_container_width=True)
        
        with st.expander("Raw Data Preview"):
            st.dataframe(df.tail(50), use_container_width=True)
        
        with st.expander("Statistics"):
            if feature_cols[0] in df.columns:
                st.write(df[feature_cols].describe())
        
    else:
        st.info("No messages received. Check Kafka connection and topic name.")
        st.code(f"Tip: Make sure Kafka is running and producing to topic '{selected_topic}'")

else:
    st.info("Click 'Refresh Data' or enable auto-refresh to load messages")

st.markdown("---")
st.caption(
    "Kafka Lab Dashboard | Industrial Equipment Monitoring | "
    f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

if auto_refresh:
    st.rerun()