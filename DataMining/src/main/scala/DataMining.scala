import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{Normalizer, VectorAssembler}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import UnsupervisedModels.KMeans
import org.apache.spark.ml.feature.MinMaxScaler


object DataMining {

  def main(args: Array[String]): Unit = {

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

    val inputElements = "src/main/resources/sample_input.txt"
    val basicTrainDF: DataFrame = ss.read.options(Map("delimiter" -> ",", "header" -> "false"))
      .csv(inputElements)

    val udf_toDouble = udf((s: String) => s.toDouble)
    val castedDF = basicTrainDF.select(udf_toDouble($"_c0"), udf_toDouble($"_c1"))

    print(basicTrainDF.count())

    val filteredDF = castedDF.filter(
      ($"_c0" =!= ""))
      .filter(
        ($"_c1" =!= ""))
      .toDF("el1", "el2")

    print(filteredDF.show())

    var assembler = new VectorAssembler()
      .setInputCols(Array("el1"))
      .setOutputCol("el1Features")

    var output = assembler.transform(filteredDF)

    var normalizer = new MinMaxScaler()
      .setInputCol("el1Features")
      .setOutputCol("normalizedEl1Features")

    var l1NormData = normalizer.fit(output).transform(output)

    assembler = new VectorAssembler()
      .setInputCols(Array("el2"))
      .setOutputCol("el2Features")

    output = assembler.transform(l1NormData)

    normalizer = new MinMaxScaler()
      .setInputCol("el2Features")
      .setOutputCol("normalizedEl2Features")

    l1NormData = normalizer.fit(output).transform(output)

    println("Normalized using L^1 norm")
    l1NormData.show(false)

    /* ======================================================= */
    /* =======================KMeans =========================== */
    /* ======================================================= */

    val combinedHotness = new KMeans(sc,l1NormData.select("normalizedEl1Features").rdd.map(_.get(0).toString().replace("[","").replace("]","").toDouble), l1NormData.select("normalizedEl2Features").rdd.map(_.get(0).toString().replace("[","").replace("]","").toDouble),5, 10)

    combinedHotness.run( "AITEEEEEEE.txt")
    sc.stop()
  }

}
