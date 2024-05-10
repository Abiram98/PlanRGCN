FROM ubuntu:22.04
COPY . /PlanRGCN
WORKDIR /PlanRGCN
RUN bash scripts/setup.sh
RUN pip3 install -r requirements2.txt
CMD bash