FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
RUN apt update\
&& apt install -y python3.11 python3-pip python3.11-dev\
&& pip install torch torchmetrics torchvision scikit_learn PyYAML tqdm diffusers transformers peft
&& apt -y autoremove
ENV PYTHONPATH="./:$PYTHONPATH"
RUN pip install vision-aided-loss torch-fidelity
