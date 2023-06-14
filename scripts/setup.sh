#!bin/bash


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install sparqlwrapper torch-geometric networkx pandas configparser matplotlib torchsummary
pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

pip3 install  dgl -f https://data.dgl.ai/wheels/repo.html
pip3 install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip3 install matplotlib
pip3 install pyvis
(cd .. && pip3 install -r requirements.txt )
pip3 install notebook
pip3 install -e ../