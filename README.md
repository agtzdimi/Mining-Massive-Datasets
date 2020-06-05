# Mining-Massive-Datasets
MSc Project: Application that detects outliers in 2D points using Spark for parallel precessing

The application .jar file is stored in the following link: https://drive.google.com/file/d/1ikgQdsOqIHPKbfkY0-rQ-7BDUTEV2rOA/view?usp=sharing

To run the .jar file an installed spark 2.4.4 version is needed. The spark version should also include the hadoop jars.

Before executing the .jar file a directory named as "spark-logs" is needed in the current working directory in order to create the log files
of the spark session.

After the creation of the "spark-logs" directory the jar can be executed:

`spark-submit --class DataMining DataMining.jar <file path>`
