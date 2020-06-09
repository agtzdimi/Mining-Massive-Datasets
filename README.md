# Mining-Massive-Datasets

MSc Project: Application that detects outliers in 2D points using Spark for parallel precessing

To execute the application a .jar file was created and uploaded in the following link: https://drive.google.com/file/d/1ikgQdsOqIHPKbfkY0-rQ-7BDUTEV2rOA/view?usp=sharing

## Prerequisites

To run the .jar file an installed spark 2.4.4 version is needed. The spark version should also include the hadoop jars

Installation steps for UNIX environments:

- `cd 'spark installation directory'`
- `wget https://archive.apache.org/dist/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz`
- `tar -zxvf spark-2.4.4-bin-hadoop2.7.tgz`
- `echo -e "export SPARK_HOME=$(pwd)/spark-2.4.4-bin-hadoop2.7\nexport PATH=\$SPARK_HOME/bin:\$PATH" >> ~/.bashrc`
- `source ~/.bashrc`
- `cd 'To the directory containing the .jar file'`

Before executing the .jar file a directory named as "spark-logs" is needed in the current working directory in order to create the log files
of the spark session.

- `mkdir spark-logs`
- `spark-submit --class DataMining DataMining.jar <file path>`
