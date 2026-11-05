# LogiEdge inference service (Task D2)
# Build from the repository root (BuildKit, default in Docker Desktop):
#   docker build --build-context pipeline=data_pipeline -t logibridge-inference:latest inference
# Layer order is chosen for OTA: everything that rarely changes comes first, the model last,
# so a model-only update rebuilds (and makes trucks download) just the final small layer.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# 1. dependencies: slim TFLite runtime, numpy, MQTT client (all pip layers before the model)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2. shared preprocessing code + frozen normalisation statistics (same files used in training)
COPY --from=pipeline preprocessing.py training_stats.npy ./

# 3. service code
COPY inference_service.py .

# 4. the model: last layer, the only one rebuilt by a model-only OTA update
COPY model.tflite .

# MODEL_PATH switches the variant without a rebuild, e.g. a bind-mounted
# /opt/logibridge/model.tflite (Task E2) or -v .../training/models:/models -e MODEL_PATH=/models/m3_pruned_ptq_int8.tflite
ENV MODEL_PATH=/app/model.tflite \
    TRUCK_ID=truck01 \
    MQTT_HOST=host.docker.internal \
    MQTT_PORT=1883 \
    UPLINK_HOST= \
    UPLINK_PORT=1883 \
    LOG_DB=/data/logibridge_log.db
VOLUME /data

CMD ["python", "inference_service.py"]
