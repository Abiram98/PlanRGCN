FROM ubuntu:22.04
COPY PlanRGCN/ /PlanRGCN/PlanRGCN/
#COPY PlanRegr/ /PlanRGCN/PlanRegr/
COPY dist_loader/ /PlanRGCN/dist_loader/
COPY feat_con_time/ /PlanRGCN/feat_con_time/
#COPY feature_extraction/ /PlanRGCN/feature_extraction/
#COPY feature_representation/ /PlanRGCN/feature_representation/
#COPY graph_construction/ /PlanRGCN/graph_construction/
COPY inductive_query/ /PlanRGCN/inductive_query/
COPY load_balance/ /PlanRGCN/load_balance/
COPY notebooks/ /PlanRGCN/notebooks/
#COPY qpe/ /PlanRGCN/qpe/
COPY qpp/ /PlanRGCN/qpp/
COPY sample_checker/ /PlanRGCN/sample_checker/
COPY scripts/ /PlanRGCN/scripts/
COPY utils/ /PlanRGCN/utils/
COPY virt_feat_conf/ /PlanRGCN/virt_feat_conf/

WORKDIR /PlanRGCN
RUN bash scripts/setup.sh
COPY requirements2.txt /PlanRGCN/requirements2.txt
COPY pp_only_qs.py /PlanRGCN/pp_only_qs.py
RUN pip3 install -r requirements2.txt
COPY .git/ /PlanRGCN/.git/
COPY .gitignore /PlanRGCN/.gitignore
RUN apt-get install git -y
#COPY .vscode/launch.json /PlanRGCN/.vscode/launch.json
#COPY .vscode/settings.json /PlanRGCN/.vscode/settings.json
COPY Dockerfile /PlanRGCN/Dockerfile
COPY .dockerignore /PlanRGCN/.dockerignore
COPY data /PlanRGCN/data
COPY run.sh /PlanRGCN/run.sh
COPY README.md /PlanRGCN/README.md
COPY test_inference_time.py /PlanRGCN/test_inference_time.py

RUN apt-get install ssh -y
RUN apt-get install rsync -y
CMD /bin/bash
