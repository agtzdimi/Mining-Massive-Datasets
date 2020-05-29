import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{MinMaxScaler, VectorAssembler}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

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
    conf.set("spark.eventLog.dir", "spark-logs")
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val sc = new SparkContext(conf)
    val ss = SparkSession.builder().master("local[*]").appName("Data Mining Project").getOrCreate()
    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames

    var inputElements = "src/main/resources/sample_input.txt"
    // var inputElements = "src/main/resources/inflated_sample.csv"
    if (args.length > 0) {
      inputElements = args(0)
    }

    val basicTrainDF: DataFrame = ss.read.options(Map("delimiter" -> ",", "header" -> "false")).csv(inputElements)

    val udf_toDouble = udf((s: String) => s.toDouble)
    val castedDF = basicTrainDF.select(udf_toDouble($"_c0"), udf_toDouble($"_c1"))

    val filteredDF = castedDF.filter(
      ($"_c0" =!= ""))
      .filter(
        ($"_c1" =!= ""))
      .toDF("el1", "el2")


    val assembler = new VectorAssembler()
      .setInputCols(Array("el1", "el2"))
      .setOutputCol("combined")
    val combined_df = assembler.transform(filteredDF)

    val normalizer = new MinMaxScaler()
      .setInputCol("combined")
      .setOutputCol("normalized_combined")

    val normalized_combined = normalizer.fit(combined_df).transform(combined_df)

    val separated_df = normalized_combined.map(f => {
      val row = f.toString().split(",")
      val el1 = row(0).replace("[", "")
      val el2 = row(1).replace("]", "")
      val el1_norm = row(4).replace("[", "")
      val el2_norm = row(5).replace("]", "")
      (el1.toDouble,
        el2.toDouble,
        el1_norm.toDouble,
        el2_norm.toDouble)
    }).withColumnRenamed("_1", "el1")
      .withColumnRenamed("_2", "el2")
      .withColumnRenamed("_3", "norm_el1")
      .withColumnRenamed("_4", "norm_el2")

    val normalized_assembler = new VectorAssembler()
      .setInputCols(Array("norm_el1", "norm_el2"))
      .setOutputCol("features")
    val training_df = normalized_assembler.transform(separated_df)

    /* ======================================================= */
    /* =======================KMeans =========================== */
    /* ======================================================= */
    //      execute KMeans clustering
    val kmeans = new KMeans().setK(5).setSeed(1L)
    val model = kmeans.fit(training_df.select("features"))
    val WSSSE = model.computeCost(training_df.select("features"))

    val clustering = model.transform(training_df)

    //      create columns for centroid points
    val clustering_ = clustering.withColumn("centroids_x", lit(0.0))
      .withColumn("centroids_y", lit(0.0))

    val centroid_clustering = clustering_.withColumn("centroids_x",
      when(col("prediction") === 0, model.clusterCenters(0)(0))
        .when(col("prediction") === 1, model.clusterCenters(1)(0))
        .when(col("prediction") === 2, model.clusterCenters(2)(0))
        .when(col("prediction") === 3, model.clusterCenters(3)(0))
        .when(col("prediction") === 4, model.clusterCenters(4)(0))
    ).withColumn("centroids_y",
      when(col("prediction") === 0, model.clusterCenters(0)(1))
        .when(col("prediction") === 1, model.clusterCenters(1)(1))
        .when(col("prediction") === 2, model.clusterCenters(2)(1))
        .when(col("prediction") === 3, model.clusterCenters(3)(1))
        .when(col("prediction") === 4, model.clusterCenters(4)(1))
    )

    // Calculate euclidean distance of each cluster element from its cluster
    // Euclidean distance function
    val euclidean = udf((x1: Double, y1: Double, x2: Double, y2: Double) => math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2)))
    val centroid_distance = centroid_clustering.withColumn("distance", euclidean(
      col("norm_el1"),
      col("norm_el2"),
      col("centroids_x"),
      col("centroids_y")))

    //      for each cluster set as the upper outlier threshold the avg(distance)+2*std(distance)
    val k = 3
    val upper_thresholds = centroid_distance.groupBy("prediction")
      .agg((avg("distance") + lit(k) * stddev("distance")).alias("upper_distance_threshold"))
      .sort(col("prediction"))
      .select("upper_distance_threshold")
      .collectAsList()
    val lower_thresholds = centroid_distance.groupBy("prediction")
      .agg((avg("distance") - lit(k) * stddev("distance")).alias("lower_distance_threshold"))
      .sort(col("prediction"))
      .select("lower_distance_threshold")
      .collectAsList()

    //  calculate upper and lower thresholds for each cluster
    val threshold_df = centroid_distance.withColumn("upper_cluster_thresholds",
      when(col("prediction") === 0, upper_thresholds.get(0)(0))
        .when(col("prediction") === 1, upper_thresholds.get(1)(0))
        .when(col("prediction") === 2, upper_thresholds.get(2)(0))
        .when(col("prediction") === 3, upper_thresholds.get(3)(0))
        .when(col("prediction") === 4, upper_thresholds.get(4)(0))
    ).withColumn("lower_cluster_thresholds",
      when(col("prediction") === 0, lower_thresholds.get(0)(0))
        .when(col("prediction") === 1, lower_thresholds.get(1)(0))
        .when(col("prediction") === 2, lower_thresholds.get(2)(0))
        .when(col("prediction") === 3, lower_thresholds.get(3)(0))
        .when(col("prediction") === 4, lower_thresholds.get(4)(0))
    )

    // filter the upper outliers
    val total_outliers_no = threshold_df.filter((col("distance") > col("upper_cluster_thresholds")) or (col("distance") < col("lower_cluster_thresholds"))).count()
    threshold_df.filter((col("distance") > col("upper_cluster_thresholds")) or (col("distance") < col("lower_cluster_thresholds"))).select(
      "el1",
      "el2").show(false)

    println(s"Total number of outliers: $total_outliers_no")

    val duration = (System.nanoTime - start) / 1e9d
    println(s"Total execution time: $duration seconds")

    sc.stop()
  }

}
