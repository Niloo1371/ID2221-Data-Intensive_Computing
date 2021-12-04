import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.feature import HashingTF, IDF, IDFModel

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import NaiveBayes

from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# maps the input into pairs of (label, sentence)
def mapper(s):
	# seperate the integer assigned to each article, each integer is a label for each category
	label = int(s.split("||")[0])
	# this seperates the news text content
	sentence = s.split("||")[1]
	return (label, sentence)

sc = SparkContext("local[2]", "pretrain")

spark = SparkSession\
		.builder\
		.appName("pretrain")\
		.getOrCreate()

trainData = []
trainLabels = []



documents = sc.textFile("train.txt")
test = sc.textFile("test.txt")

schema = StructType([
StructField("label", IntegerType(), True),
StructField("sentence", StringType(), True)])

# create dataframe for training data, DataFrame[label: int, sentence: string]
df = spark.createDataFrame(documents.map(mapper), schema)
testDf = spark.createDataFrame(test.map(mapper), schema)

# tokenize the sentence (seperate the sentence into words, DataFrame[label: int, sentence: string, words: array<string>]
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
tokenized = tokenizer.transform(df)

# remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
removed = remover.transform(tokenized)

# tf-idf, generate frequency vectors, based on the documentation of ML features: https://spark.apache.org/docs/latest/ml-features
# take the sentences (bag of words) and convert them into fixed-length feature vectors.
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=300)
featurizedData = hashingTF.transform(removed)

# fit on the dataset and produces an IDFModel. Take feature vectors and rescale them
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)

# classifier
nb = NaiveBayes(smoothing=1.2, modelType="multinomial")

# build pipeline using tokenizer, remover, hashing, idf and NaiveBayes model, reference of ML pipeline is here: https://spark.apache.org/docs/latest/ml-pipeline.html#pipeline 
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idfModel, nb])


# train model by pipeline
model = pipeline.fit(df)
# save the model
try:
	model.save("pipeline_model.model")
except:
	model.write().overwrite().save("pipeline_model.model")

# load the saved model 
modelLoad = PipelineModel.load("pipeline_model.model")
# make prediction
predictions = modelLoad.transform(testDf)
predictions.select("prediction", "label").show(40)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
print("Accuracy of Naive Bayes")
accuracy = evaluator.evaluate(predictions)
print(accuracy)
print("Test Error = %g " % (1.0 - accuracy))

