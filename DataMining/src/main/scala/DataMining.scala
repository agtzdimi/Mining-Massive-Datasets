import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{Normalizer, VectorAssembler}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{max, min, avg, stddev}
import org.apache.spark.sql.functions.{col, lit, when}
import org.apache.spark.sql.Row
//import UnsupervisedModels.KMeans
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.linalg.Vector
import math._


import org.apache.spark.ml.feature.MinMaxScaler


object DataMining {

   def main(args: Array[String]): Unit = {
      val start = System.nanoTime

      val conf = new SparkConf().setMaster("local[*]")
                                .setAppName("Data Mining Project").setSparkHome("src/main/resources")
      conf.set("spark.driver.memory", "14g")
      conf.set("spark.hadoop.validateOutputSpecs", "false")
      conf.set("spark.executor.instances", "1")
      conf.set("spark.executor.cores", "8")
      conf.set("spark.executor.memory", "2g")
      conf.set("spark.cores.max", "8")
      conf.set("spark.eventLog.enabled", "true")
      conf.set("spark.eventLog.dir", "src/main/resources/spark-logs")
      Logger.getLogger("org").setLevel(Level.OFF)
      Logger.getLogger("akka").setLevel(Level.OFF)
      val sc = new SparkContext(conf)
      val ss = SparkSession.builder().master("local[*]").appName("Data Mining Project").getOrCreate()
      import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

//      set input file
//      val inputElements = "src/main/resources/sample_input.txt"
      val inputElements = "src/main/resources/inflated_sample.csv"
      if (args.size > 0) {val inputElements = args(0)}


      val basicTrainDF: DataFrame = ss.read.options(Map("delimiter" -> ",", "header" -> "false")).csv(inputElements)

      val udf_toDouble = udf((s: String) => s.toDouble)
      val castedDF = basicTrainDF.select(udf_toDouble($"_c0"), udf_toDouble($"_c1"))

      val filteredDF = castedDF.filter(
         ($"_c0" =!= ""))
                               .filter(
                                  ($"_c1" =!= ""))
                               .toDF("el1", "el2")

      var assembler = new VectorAssembler()
         .setInputCols(Array("el1", "el2"))
         .setOutputCol("combined")

      var combined = assembler.transform(filteredDF)

      var normalizer = new MinMaxScaler()
         .setInputCol("combined")
         .setOutputCol("normalized_combined")

      var normalized_combined = normalizer.fit(combined).transform(combined)

      /* ======================================================= */
      /* =======================KMeans =========================== */
      /* ======================================================= */
      val el1 = normalized_combined.select("normalized_combined").rdd.map(x => x.toString().split(",")(0)
                                                                                .replace("[", "")
                                                                                .toDouble)
      val el2 = normalized_combined.select("normalized_combined").rdd.map(x => x.toString().split(",")(1)
                                                                                .replace("]", "")
                                                                                .toDouble)
      val normalized_seq: RDD[(Double, Double)] = el1 zip el2
      var normalized_df = normalized_seq.toDF("el1", "el2")
      var normalized_assembler = new VectorAssembler()
         .setInputCols(Array("el1", "el2"))
         .setOutputCol("features")
      var training = normalized_assembler.transform(normalized_df)

      //      execute KMeans clustering
      println("Executing KMeans clustering")
      val kmeans = new KMeans().setK(5).setSeed(1L)
      val model = kmeans.fit(training.select("features"))
      val WSSSE = model.computeCost(training.select("features"))
      println("WSSSE", WSSSE)

      val clustering = model.transform(training)

      //      euclidean distance function
      val euclidean = udf((x1: Double, y1: Double, x2: Double, y2: Double) => math.sqrt(math.pow((x1 - y1), 2) + math.pow((x2 - y2), 2)))
      //      create columns for centoid points
      val clustering_ = clustering.withColumn("centroids_x", lit(0.0))
                                  .withColumn("centroids_y", lit(0.0))

      val centroid_clustering = clustering_.withColumn("centroids_x", when(col("prediction") === 0, model.clusterCenters(0)(0))
         .when(col("prediction") === 1, model.clusterCenters(1)(0))
         .when(col("prediction") === 2, model.clusterCenters(2)(0))
         .when(col("prediction") === 3, model.clusterCenters(3)(0))
         .when(col("prediction") === 4, model.clusterCenters(4)(0))
      ).withColumn("centroids_y", when(col("prediction") === 0, model.clusterCenters(0)(1))
         .when(col("prediction") === 1, model.clusterCenters(1)(1))
         .when(col("prediction") === 2, model.clusterCenters(2)(1))
         .when(col("prediction") === 3, model.clusterCenters(3)(1))
         .when(col("prediction") === 4, model.clusterCenters(4)(1))
      )
      //      calculate eclidean distance of each cluster element from its cluster centoid
      var centroid_distance = centroid_clustering.withColumn("distance", euclidean(
         col("el1"),
         col("el2"),
         col("centroids_x"),
         col("centroids_y")))

      //      for each cluster set as the outlier threshold the avg(distance)+2*std(distance)
      var thresholds = centroid_distance.groupBy("prediction")
                                        .agg((avg("distance") + lit(2) * stddev("distance") ).alias("distance_threshold"))
                                        .sort(col("prediction"))
                                        .select("distance_threshold")
                                        .collectAsList()

      val threshold_df = centroid_distance.withColumn("cluster_thresholds", when(col("prediction") === 0, thresholds.get(0)(0))
         .when(col("prediction") === 1, thresholds.get(1)(0))
         .when(col("prediction") === 2, thresholds.get(2)(0))
         .when(col("prediction") === 3, thresholds.get(3)(0))
         .when(col("prediction") === 4, thresholds.get(4)(0))
      )

      //      filter the ouliers
      var ouliers_no = threshold_df.filter(col("distance") > col("cluster_thresholds")).count()
      println(s"Total number of outliers: $ouliers_no")
      threshold_df.filter(col("distance") > col("cluster_thresholds")).select(
         "el1",
         "el2",
         "distance",
         "cluster_thresholds",
         "prediction").show()

      val duration = (System.nanoTime - start) / 1e9d
      println(s"Total execution time: $duration seconds")

      sc.stop()
   }

}
