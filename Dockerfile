FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
RUN apt update\
&& apt install -y python3.10 python3-pip python3.10-dev\
&& pip install torch==2.3.1 torchmetrics==1.4.0.post0 torchvision==0.18.1 scikit_learn==1.5.0 PyYAML==6.0.1 tqdm==4.66.4\
&& apt -y autoremove
