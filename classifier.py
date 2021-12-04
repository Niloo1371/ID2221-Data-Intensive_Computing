import findspark
findspark.init()
import json
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from elasticsearch import Elasticsearch


brokers, topic = sys.argv[1:]


def connect_elasticsearch():
    es = None
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    return es

def getSparkSessionInstance(sparkConf):
    if ('sparkSessionSingletonInstance' not in globals()):
        globals()['sparkSessionSingletonInstance'] = SparkSession\
            .builder\
            .config(conf=sparkConf)\
            .getOrCreate()
    return globals()['sparkSessionSingletonInstance']


def process(rdd):
    try:
        spark = getSparkSessionInstance(rdd.context.getConf())
        rowRdd = rdd.map(lambda w: Row(label= int(w[1].split("||")[0]), sentence= w[1].split("||")[1]))
        
        wordsDataFrame = spark.createDataFrame(rowRdd)

        # Load pretrain model
        modelLoad = PipelineModel.load("Desktop/big data/hw3/pipeline_model.model")
        predictions = modelLoad.transform(wordsDataFrame)
        print("label")
        print(predictions.select("label").collect())
        print(predictions.select("prediction").collect())
        print("prediction")
        es = connect_elasticsearch()
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("accuracy")
        data = json.dumps({"label": predictions.select("label").collect(), "prediction": predictions.select("prediction").collect()})
        es.index(index = "index", doc_type="type", body = data)
    except:
        pass


# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext("local[2]", "classifier")
ssc = StreamingContext(sc, 5)
# Create a DStream that will connect to Kafka
lines = KafkaUtils.createDirectStream(ssc, [topic],{"metadata.broker.list": brokers})
lines.foreachRDD(process)

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate