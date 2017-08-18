package com.sparkdemo.diabetes.streaming

import java.util.HashMap

import org.apache.kafka.clients.producer.{KafkaProducer, ProducerConfig, ProducerRecord}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.streaming.{Minutes, Seconds, StreamingContext}
import org.apache.spark.streaming.kafka.KafkaUtils

object ReadmissionPredictor {

    def setupLogging() = {
        import org.apache.log4j.{Level, Logger}
        val rootLogger = Logger.getRootLogger()
        rootLogger.setLevel(Level.ERROR)
    }
    def main(args: Array[String]) {
        if (args.length < 5) {
            System.err.println("Usage: ReadmissionPredictor <zkQuorum> <brokers> <group> <TOPICS> <numThreads>")
            System.exit(1)
        }
        val modelbasepath = "/Users/ka3862/Documents/demo/model"


        val Array(zkQuorum, brokers, group, topics, numThreads) = args

        val sparkConf = new SparkConf().setAppName("ReadmissionPredictor").setMaster("local[*]")
        val ssc = new StreamingContext(sparkConf, Seconds(2))
        ssc.checkpoint("checkpoint")

        setupLogging()

        // Zookeeper connection properties
        val props = new HashMap[String, Object]()
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, brokers)
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,
            "org.apache.kafka.common.serialization.StringSerializer")
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,
            "org.apache.kafka.common.serialization.StringSerializer")

        val topicMap = topics.split(",").map((_, numThreads.toInt)).toMap
        val lines = KafkaUtils.createStream(ssc, zkQuorum, group, topicMap).map(_._2)

        val vectors = lines.map(s => Vectors.dense(s.split(',').map(_.toDouble)))

        val producer = new KafkaProducer[String, String](props)
        val kmeansModel = KMeansModel.load(ssc.sparkContext, modelbasepath + "/kmeans")

        println ("*** Number of KMeans clusters ***")
        println ( kmeansModel.clusterCenters.length)
        println ( kmeansModel.clusterCenters(0).toString)

        val randomForestModel = RandomForestModel.load( ssc.sparkContext,  modelbasepath + "/randomforest")

        println ("** Number of trees ***")
        println (randomForestModel.numTrees)

        // TODO: Switch to using mapPartitions instead of creating a producer for each record
        val pred = vectors.map( v =>  {
            val producer = new KafkaProducer[String, String](props)
            val pred = kmeansModel.predict(v).toInt
            val message = new ProducerRecord[String, String]("KMEANS", pred.toString, v.toString)
            producer.send(message)

            producer.close()
            pred


        } )

        // TODO: Switch to using mapPartitions instead of creating a producer for each record
        val rfPred = vectors.map( v =>  {
            val producer = new KafkaProducer[String, String](props)
            println (randomForestModel.numTrees)
            val pred = randomForestModel.predict(v).toInt
            val topic = "RF"
            val message = new ProducerRecord[String, String](topic, pred.toString, v.toString)
            producer.send(message)
            producer.close
            pred
        } )

        pred.print()
        rfPred.print()
        ssc.start()
        ssc.awaitTermination()
    }
}
