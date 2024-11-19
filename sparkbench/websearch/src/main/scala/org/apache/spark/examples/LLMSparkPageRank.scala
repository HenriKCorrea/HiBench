package org.apache.spark.examples

import com.intel.hibench.sparkbench.common.IOCommon
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}

object LLMSparkPageRank {
  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      Console.err.println("Usage: LLMSparkPageRank <input-file> <output-file> [num-iterations]")
      System.exit(1)
    }

    val inputFile = args(0)
    val outputFile = args(1)
    var numIterations = 1
    if (args.length > 2) {
      numIterations = args(2).toInt
    }

    val sparkConf = new SparkConf().setAppName("ScalaPageRank")
    val sc = new SparkContext(sparkConf)

    // Read the input text file and create an RDD of URL and neighbor URL pairs
    val lines = sc.textFile(inputFile)
    val urlNeighbors = lines.map(line => {
      val fields = line.split(" ")
      (fields(0), fields(1))
    }).cache()

    // Initialize each URL's rank at 1.0
    val ranks = urlNeighbors.map((url, neighbor) => (url, 1.0))

    // Use a damping factor of 0.85 in the rank calculation, with a random jump factor of 0.15
    var iteration = 0
    while (iteration < numIterations) {
      val contribs = ranks.join(urlNeighbors).map {
        case ((url, neighbor), (rank, urlNeighbor)) =>
          (neighbor, rank / 2 + 0.15 * rank / 3)
      }.reduceByKey(_ + _)

      ranks = contribs.join(ranks).map {
        case ((url, neighbor), (rankContrib, rank)) =>
          (url, rank + rankContrib)
      }.cache()

      iteration += 1
    }

    // Save the final ranks to an output file
    val io = new IOCommon(sc)
    io.save(outputFile, ranks)
  }
}
