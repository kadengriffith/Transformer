FROM tensorflow/tensorflow:2.0.0b1-gpu-py3

WORKDIR /home

RUN pip install --upgrade pip
RUN pip install --upgrade virtualenv
RUN virtualenv /home
RUN . /home/bin/activate
RUN pip install tensorflow-datasets