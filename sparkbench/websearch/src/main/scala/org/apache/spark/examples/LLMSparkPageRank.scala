package org.apache.spark.examples
import com.intel.hibench.sparkbench.common.IOCommon
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
/**
 * Computes the PageRank of URLs from an input file. Input file should
 * be in format of:
 * URL         neighbor URL
 * URL         neighbor URL
 * URL         neighbor URL
 * ...
 * where URL and their neighbors are separated by space(s).
 */
object LLMSparkPageRank {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Usage: LLMSparkPageRank <input_file> <output_filename> [<iter>]")
      System.exit(1)
    }
    val sparkConf = new SparkConf().setAppName("ScalaPageRank")
    val input_path = args(0)
    val output_path = args(1)
    val iters = if (args.length > 2) args(2).toInt else 10
    val ctx = new SparkContext(sparkConf)
    // Load the input file and map each line to a pair of URLs
    val links = ctx.textFile(input_path).map(line => {
      val parts = line.split("\\s+")
      (parts(0), parts(1))
    })
    // Create a set of all unique URLs
    val urls = links.flatMap(link => Iterator(link._1, link._2)).distinct()
    // Initialize the ranks for each URL to 1.0
    var ranks = urls.map(url => (url, 1.0)).reduceByKey((a, b) => a)
    // Run the PageRank algorithm for the specified number of iterations
    for (i <- 0 until iters) {
      // Calculate the number of links for each URL
      val numLinks = links.map(link => (link._1, 1)).reduceByKey((a, b) => a + b)
      // Join the links with the ranks and the number of links
      val contribs = links.join(ranks).join(numLinks).map {
        case (url, ((link, rank), numLink)) =>
          (link, rank / numLink)
      }
      // Sum up the contributions from all URLs
      ranks = contribs.reduceByKey((a, b) => a + b).mapValues(value => 0.15 + 0.85 * value)
    }
    val io = new IOCommon(ctx)
    io.save(output_path, ranks)
    ctx.stop()
  }
}