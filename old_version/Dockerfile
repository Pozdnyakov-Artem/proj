FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ../requirements.txt ./

RUN pip install --default-timeout=1000 -r requirements.txt

COPY test.py detect_with_zone.py db_worker.py models.py save_img_worker.py ./

ENV PYTHONUNBUFFERED=1

CMD ["python", "detect_with_zone.py"]
#CMD ["python", "test.py"]