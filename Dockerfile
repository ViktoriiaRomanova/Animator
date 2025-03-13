FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
RUN apt update
COPY requirements.txt .
RUN apt install -y python3.10 python3-pip python3.10-dev
RUN pip install -r requirements.txt
RUN apt -y autoremove
WORKDIR /workspace
RUN chmod 777 /workspace
#RUN mkdir -m 777 .cache
#ENV PYTHONPATH="./:$PYTHONPATH"
#COPY --chmod=777 tmpfolder/ .cache/
COPY trained_models/segmentation/99.pt /workspace/trained_models/segmentation/