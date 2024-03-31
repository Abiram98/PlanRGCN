#!bin/bash
apt-get update && apt-get upgrade -y

apt-get install python3 -y
apt-get install python3-pip -y
apt-get install tmux -y

apt install libcairo2-dev pkg-config python3-dev -y
pip3 install -r /PlanRGCN/requirements2.txt

#java installation
apt-get -y install maven
apt-get -y install openjdk-17-jdk openjdk-17-jre
mvn package -f "/PlanRGCN/PlanRGCN/qpe/pom.xml"
mvn install:install-file -Dfile=/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar -DpomFile=/PlanRGCN/qpe/pom.xml

apt-get install graphviz graphviz-dev -y