FROM nvcr.io/nvidia/tensorflow:23.03-tf1-py3
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN apt update || true
RUN apt install -y python3-tk
RUN pip3 install sacred pytest
RUN git config --global --add safe.directory /workspace
RUN pip3 install ipython ipdb
