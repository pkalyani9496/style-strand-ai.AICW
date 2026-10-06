FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY .streamlit/config.toml .streamlit/config.toml
COPY app/ app/

COPY models/hair_presence_v2/hair_presence_mobilenetv2_v2.keras models/hair_presence_v2/
COPY models/hair_presence_v2/class_names.json models/hair_presence_v2/
COPY models/hair_type/hair_type_mobilenetv2.keras models/hair_type/
COPY models/hair_type/class_names.json models/hair_type/
COPY models/hair_segmentation/hair_segmentation_unet_best.keras models/hair_segmentation/
COPY models/hair_condition_gate/best_hair_condition_gate_v2.keras models/hair_condition_gate/
COPY models/hair_condition_gate/class_names.json models/hair_condition_gate/
COPY models/hair_disease/best_hair_disease_model.keras models/hair_disease/
COPY models/hair_disease/class_names.json models/hair_disease/

EXPOSE 7860

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false"]
