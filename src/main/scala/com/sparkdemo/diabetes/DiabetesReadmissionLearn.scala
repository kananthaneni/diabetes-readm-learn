package com.sparkdemo.diabetes

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

import scala.util.Try

object DiabetesReadmissionLearn {

  def main(args: Array[String]) {
    if (args.length < 1) {
          System.err.println("Usage: DiabetesReadmissionLearn <modelBasePath")
          System.exit(1)
    }

    val modelBasePath = args(0)

    val sparkSession = SparkSession.builder
      .master("local[*]")
      .appName("ReadmissionLearn")
      .getOrCreate()

    import sparkSession.implicits._

    val categoricalCols = List("admission_type_id", "race", "gender", "age", "diabetesMed", "insulin", "change", "discharge_disposition_id")
    val numericCols = List("num_medications", "num_procedures", "num_lab_procedures", "number_outpatient",
      "number_emergency", "number_inpatient", "number_diagnoses", "time_in_hospital")

    val colstoUse = (List("readmitted") ::: categoricalCols ::: numericCols )

    // udf to filter out non numeric data
    val isNumeric = udf((s: String) => Try(s.toDouble).isSuccess)

    /**
      * Builds an array by one hot encoding the value using the possible values provided
      */
    def buildArray(possibleVals: Array[String]) = udf(( value: String) => {
      var strBuilder = StringBuilder.newBuilder
      val buff =  scala.collection.mutable.ArrayBuffer.empty[String]
      possibleVals.map( row => {
        buff += (if (row.equals(value)) "1" else "0")
      })

      buff
    })

    /**
      * Returns true if data is present. or false if it is empty or '?'
      */
    val isDataPresent = udf((s:  String) => !(s.equals("?") || s.trim.equals("")))

    val sparkContext = sparkSession.sparkContext

    //read input CSV file with header and infer schema
    val df = sparkSession.read.option("header", "true").option("inferSchema", "false").csv("src/main/resources/diabetic_data.csv")

    // Get only columns we are interested in
    var trimmedDF = df.select(colstoUse.head, colstoUse.tail: _*)

    // Filter out rows missing data in the columns we care about
    for (strCol: String <- categoricalCols) {
      trimmedDF = trimmedDF.filter( isDataPresent (trimmedDF.col(strCol)))
    }
    trimmedDF.printSchema()

    // For categorical colummns create a Map with column name as the key and the array of possible values as value
    var categoricalColMap: scala.collection.mutable.Map[String, Array[String]] = scala.collection.mutable.Map[String, Array[String]]()
    for (categoricalCol :String <- categoricalCols) {
      categoricalColMap (categoricalCol) = trimmedDF.select(categoricalCol).distinct().collect().map(_.getAs[String](0))
    }

    // TODO: This seems very inefficient..
    // Is it more efficient to just use RDDs and use broadcast variables to send the categorical values to the Map?
    for (categoricalCol :String <- categoricalCols) {
      val refData: Array[String] = categoricalColMap(categoricalCol)

      // First add all the values for the category as an array to a column and drop the original column
      trimmedDF= trimmedDF.withColumn("udf_"+categoricalCol,  buildArray(refData)(trimmedDF(categoricalCol))).drop(categoricalCol)
      // Go through the array and add each one a separate column
      for (i <- 0 to categoricalColMap(categoricalCol).length - 1)
      {
        trimmedDF = trimmedDF.withColumn(categoricalCol+'_'+i, trimmedDF.col("udf_"+categoricalCol)(i))
      }

      // Remove the column with the array
      trimmedDF = trimmedDF.drop("udf_"+categoricalCol)
    }

    // Persist since we are using the same data set for transformations for KMeans and Decision Trees
    trimmedDF.persist()

    kmeansLearn( trimmedDF, sparkSession, modelBasePath)

    // Convert readmitted to numeric
    val data = trimmedDF.withColumn("readmitted", readmittedToNumeric($"readmitted"))
    randomForestLearn (data, sparkSession, modelBasePath)

    sparkSession.close()

  }

  // Based off of Spark examples
  def kmeansLearn (df:DataFrame, sparkSession: SparkSession, modelBasePath: String ) {

    val kmeansDataPath = modelBasePath + "/data/kmeans"
    df.drop("readmitted").write.option("header", "false").csv( kmeansDataPath)
    val sc = sparkSession.sparkContext

    // TODO: This is a quick hack to get the program going. Should avoid writing and reading
    // Load and parse the data
    val data = sc.textFile(kmeansDataPath)
    val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 3
    val numIterations = 30
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    // TODO: Try increasing the clusters and check the sum of squared errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model
    clusters.save(sc, modelBasePath + "/model/kmeans")
  }


  /**
    * Converts the Row into LabeledPoint
    */
  def toLabeledPoint (r: Row): LabeledPoint = {
    val dbls:Seq[Double] = r.toSeq.map(_.toString.toDouble)
    val lbl = dbls(0)
    val vals = dbls.slice(1, dbls.length)
    LabeledPoint(lbl, Vectors.dense(vals.toArray))
  }

  // Map of readmissions predictions to it's numeric value
  val predMap = Map("NO" -> 0, ">30" -> 1,  "<30" -> 2)

  // Convert readmitted to Numeric
  val readmittedToNumeric = udf((s: String) => predMap(s))
  /**
    * Writes data out in LibSVM format to be used by decision trees
    *
    * @param df
    * @param sc
    */
  def writeAsLibSVMFile (df:DataFrame, sc:SparkContext, modelBasePath: String) : Unit = {

    val parsedData2 = df.rdd.map( toLabeledPoint(_)).cache()
    MLUtils.saveAsLibSVMFile(parsedData2, modelBasePath + "/data/dectrees")

  }

  // Based off of Spark examples
  def randomForestLearn(df:DataFrame, sparkSession: SparkSession, modelBasePath: String ) {
    val sc = sparkSession.sparkContext
    val dataPath =  modelBasePath + "/data/randomForest"

    val labeledData = df.rdd.map( toLabeledPoint(_)).cache()
    MLUtils.saveAsLibSVMFile(labeledData, dataPath)

    // TODO: Remove the Hack and try using Pipeline
    val data = MLUtils.loadLibSVMFile(sparkSession.sparkContext, dataPath)

    // Use 80% for training and 20% for testing
    val splits = data.randomSplit(Array(0.8, 0.2))
    val (trainingData, testData) = (splits(0), splits(1))

    val numberOfClasses = 3
    val numberOfTrees = 25
    val featureSubsetStrategy = "auto"
    val impurity = "entropy"
    val maxDepth = 20
    val maxBins = 300

    // Train model on training data
    val model = RandomForest.trainClassifier(trainingData, numberOfClasses,  Map[Int, Int](),
      numberOfTrees, "auto", impurity, maxDepth, maxBins)

    // Evaluate model on test instances
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    def getMetrics(model: RandomForestModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
      val predictionsAndLabels = data.map(labelPoint =>
        (model.predict(labelPoint.features), labelPoint.label)
      )
      val metrics = new MulticlassMetrics(predictionsAndLabels)

      metrics
    }

    // Save and load model
    model.save(sc, modelBasePath + "/model/randomforest")

    val evaluations =
      for (impurity <- Array("entropy", "gini"); depth <- Array(20); bins <- Array(100, 300))
        yield {
          val model = RandomForest.trainClassifier(trainingData, numberOfClasses, Map[Int, Int](),
            numberOfTrees, featureSubsetStrategy, impurity, depth, bins)

          val testAccuracy = getMetrics(model, testData).accuracy
          ((impurity, depth, bins), (testAccuracy))
        }

    evaluations.sortBy(_._2).reverse.foreach(println)
  }
}