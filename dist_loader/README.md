# Distance Loader
The goal of this code is 
1) verify that all distances have been accounted for and 
2) to calculate the time spent on the distance matrix for the training data specifically.

```
mvn package -f pom.xml 
java -jar -Xms80G -Xmx100G target/dist_loader-1.0-SNAPSHOT.jar 
java -jar -Xms20G -Xmx20G target/dist_loader-1.0-SNAPSHOT.jar te
```