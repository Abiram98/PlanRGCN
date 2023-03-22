FROM ubuntu:22.10

RUN apt-get update

RUN apt install python3.10
RUN apt install python3-pip -y
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install sparqlwrapper
