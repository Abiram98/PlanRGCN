#!bin/bash
apt-get update && apt-get upgrade -y

apt-get install python3 -y
apt-get install python3-pip -y
apt-get install tmux -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install sparqlwrapper torch-geometric networkx pandas configparser matplotlib torchsummary
pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

pip3 install  dgl -f https://data.dgl.ai/wheels/repo.html
pip3 install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip3 install matplotlib
pip3 install pyvis
pip3 install pydot
(cd .. && pip3 install -r requirements.txt )
pip3 install notebook
#pip3 install -e ../
pip3 install -e feature_extraction
pip3 install -e graph_construction
pip3 install -e dgl_classifier 
pip3 install -U "ray[data,train,tune,serve]"
#java installation
apt-get -y install maven
apt-get -y install openjdk-17-jdk openjdk-17-jre
#mvn exec:java -f "/PlanRGCN/qpe/pom.xml"


apt-get install graphviz graphviz-dev -y
pip install pygraphviz

pip3 install -e feature_representation/
pip3 install -e trainer
pip3 install -e sample_checker 
pip3 install -e utils/
pip3 install -e load_balance/

pip3 install xgboost
pip3 install xgboost_ray
pip3 install -e PlanRegr/
pip3 install pytz