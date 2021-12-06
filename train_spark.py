import awswrangler as wr
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import row

def main():
    
    lines = spark.read.text(s3_input_data).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]),
                                        movieId=int(p[1]),
                                        rating=float(p[2]),
                                        timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    (traning, test) = ratings.randomSplit([0.8,0.2])
    
    # build recommendation model using A:S on the training data
    als = ALS(maxIter =5,
             regParam=0.01,
             userCol="userId",
             itemCol = "itemId",
             ratingCol= "raint",
             coldStartStrategy="drop")
    model = als.fit(training)
    
    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName='rmse',
                                   labelCol='rating',
                                   predictionCol = 'prediction')
    rmse = evaluator.evaluate(predictions)
    
    # Generate top 10 recommendations for each user
    
    userRecs = model.recommendForAllUsers(10)
    userRecs.show()
    
    # Write top 10 recommendations for each user
    
    userrecs.repartition(1).write.mode('overwrite')\
    .option('header',True).option('delimiter','\t')\
    .csv(f'{s3_output_data}/recommendations')