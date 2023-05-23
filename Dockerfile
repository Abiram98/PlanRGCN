FROM ubuntu:22.10

RUN apt-get update && apt-get upgrade -y

RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install sparqlwrapper torch-geometric networkx pandas configparser matplotlib torchsummary
RUN pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
COPY scripts/setup.sh setup.sh
RUN bash setup.sh
