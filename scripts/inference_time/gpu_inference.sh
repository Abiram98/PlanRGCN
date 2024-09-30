apt install python3.10-venv
python3 -m venv torch
source torch/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#pip3 install dgl==1.1.3
pip3 install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip3 install json5==0.9.14
pip3 install matplotlib==3.8.2
pip3 install networkx==3.0
pip3 install notebook==7.0.6
pip3 install pandas==2.1.4
pip3 install rdflib==7.0.0
pip3 install scikit-learn==1.3.2
pip3 install SPARQLWrapper==2.0.0
pip3 install tensorboardX==2.6.2.2
pip3 install tensorflow