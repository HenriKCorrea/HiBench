package com.intel.hibench.sparkbench.ml
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import scopt.OptionParser
object LLMNaiveBayesExample {
  case class Params(
      input: String = null,
      lambda: Double = 1.0)
  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("LLMNaiveBayesExample") {
      head("LLMNaiveBayesExample: an example Naive Bayes app for parquet dataset.")
      opt[Double]("lambda")
        .text(s"lambda (smoothing constant), default: ${defaultParams.lambda}")
        .action((x, c) => c.copy(lambda = x))
      arg[String]("<input>")
        .text("input paths to labeled examples in parquet format")
        .required()
        .action((x, c) => c.copy(input = x))
    }
    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      sys.exit(1)
    }
  }
  def run(params: Params) {
    val spark = SparkSession
      .builder
      .appName(s"LLMNaiveBayesExample with $params")
      .getOrCreate()
    val df = spark.read.parquet(params.input)
    // Split the data into training and test sets (20% held out for testing)
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2), seed = 1234L)
    // Train a NaiveBayes model.
    val model = trainNaiveBayesModel(trainingData)
    // Select example rows to display.
    val predictions = model.transform(testData)
    predictions.show()
    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy = $accuracy")
    spark.stop()
  }
  /**
   * Trains a Naive Bayes model using the provided training data.
   *
   * @param trainingData The DataFrame containing the training data.
   * @return The trained Naive Bayes model.
   */
  def trainNaiveBayesModel(trainingData: DataFrame): NaiveBayesModel = {
    val numClasses = trainingData.select("label").distinct().count().toInt
    val numFeatures = trainingData.columns.length - 1 // exclude the label column
    val classCounts = trainingData
      .groupBy("label")
      .agg(count("*").alias("classCount"))
      .withColumnRenamed("count(1)", "classCount")
      .cache()
    val featureColumns = trainingData.columns.filter(_ != "label").map(c => sum(col(c).cast("long")).alias(c))
    val featureCounts = trainingData
      .groupBy("label")
      .agg(featureColumns.head, featureColumns.tail: _*)
      .cache()
    val priorProbabilities = classCounts
      .withColumn(
        "priorProbability",
        col("classCount") / lit(trainingData.count())
      )
      .cache()
    val conditionalProbabilities = featureCounts
      .crossJoin(broadcast(priorProbabilities))
      .withColumn(
        "conditionalProbability",
        lit(1) + col("classCount") / trainingData.count()
      )
      .cache()

    val model = new NaiveBayesModel(
      numClasses = numClasses,
      numFeatures = numFeatures,
      priorProbabilities = priorProbabilities,
      conditionalProbabilities = conditionalProbabilities
    )
    model
  }
  class NaiveBayesModel(
      numClasses: Int,
      numFeatures: Int,
      priorProbabilities: DataFrame,
      conditionalProbabilities: DataFrame
  ) extends Serializable {
    def transform(testData: DataFrame): DataFrame = {
    // calculate probabilities for each class and select the class with maximum probability
    val predictions = testData
      .crossJoin(broadcast(priorProbabilities))
      .withColumn(
        "logLikelihood",
        exp(
          testData.columns
            .filter(_ != "label")
            .map(c =>
              log(col(c).cast("double"))
            )
            .reduce(_ + _) + log(col("priorProbability").cast("double"))
        )
      )
      .select("label", "logLikelihood")
      .sort(desc("logLikelihood"))
      .limit(1)
        predictions
    }
  }
}