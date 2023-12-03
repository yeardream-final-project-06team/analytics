FROM tensorflow/tensorflow:1.15.5-gpu-py3

COPY . /analytics

RUN pip3 install -r /analytics/requirements.txt