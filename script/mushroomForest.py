from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import pandas as pd
import numpy as np




if __name__=='__main__':

    sc = SparkContext()

    spark = SparkSession.builder \
        .appName('Mushroom') \
        .getOrCreate()

    mushrooms = spark.read.csv('kaggle-mushroom/input/mushrooms.csv', header=True)

    in_cols = mushrooms.schema.names[1:]

    str_indexers = [StringIndexer(inputCol=c, outputCol=c+'_idx') for c in in_cols]
    # a list of StringIndexer objects to convert strings to integer indices
    # each indexer is responsible for converting one feature column

    onehot_encoders = [OneHotEncoder(dropLast=False, inputCol=c+'_idx', outputCol=c+'_onehot') for c in in_cols]
    # a list of OneHotEncoder objects to convert integer indices of cat levels to one-hot encoded columns
    # each encoder is responsible fore encoding one feature column

    onehot_cols = [c+'_onehot' for c in in_cols]

    feat_assembler = VectorAssembler(inputCols=onehot_cols, outputCol='features')
    # a VectorAssembler object that assembles all the one-hot encoded columns into one column,
    # each row of which is a vector of all the numbers in those one-hot columns.
    # e.g.
    # +-----+-----+-----+-----+---------------------+
    # |cat_0|cat_1|cat_2|cat_3|             features|
    # +-----+-----+-----+-----+---------------------+
    # |    1|    0|    0|    0| [1.0, 0.0, 0.0, 0.0]|
    # |    0|    1|    0|    0| [0.0, 1.0, 0.0, 0.0]|
    # |    0|    0|    0|    1| [0.0, 0.0, 0.0, 1.0]|
    # +-----+-----+-----+-----+---------------------+

    label_indexer = StringIndexer(inputCol=mushrooms.schema.names[0], outputCol='poisonous')
    # a StringIndexer object that converts <class> column's {e, p} to {0, 1}
    # Because there are more 'e' class in the sample, it will be encoded 0, since StringIndexer gives more frequent levels a lower index
    # Run `mushrooms.groupby('class').count().show()` in pyspark shell to see counts of each class

    pipeline = Pipeline(stages=str_indexers+onehot_encoders+[feat_assembler, label_indexer])
    # A Pipeline object that combines all the transformations we defined above.

    # Use the pipeline object to transform our dataframe
    mushrooms_trans = pipeline \
                        .fit(mushrooms) \
                        .transform(mushrooms) \
                        .cache()

    # Train-test split
    mushrooms_train, mushrooms_val = mushrooms_trans.randomSplit([0.7, 0.3], seed=2017)

    model = RandomForestClassifier(labelCol='poisonous', featuresCol='features', numTrees=200) \
            .fit(mushrooms_train)

    pred = model.transform(mushrooms_val)



    results = pred.select(['probability', 'prediction', 'poisonous']).collect()
    # Select the columns relevant for evaluation
    # `results` looks like this before .collect():
    # +--------------------+----------+---------+
    # |         probability|prediction|poisonous|
    # +--------------------+----------+---------+
    # |[0.97024593961675...|       0.0|      0.0|
    # |[0.96303265951929...|       0.0|      0.0|
    # |[0.95909221894651...|       0.0|      0.0|
    # |[0.95958294573868...|       0.0|      0.0|
    # |[0.95580449199223...|       0.0|      0.0|
    # +--------------------+----------+---------+
    #
    # After .collect(), `results` become a list of Row objects

    results_prob_pred = [(float(row[0][0]), float(row[1])) for row in results]
    prob_pred = sc.parallelize(results_prob_pred)
    score = BinaryClassificationMetrics(prob_pred)
    print('AUC ROC score:', score.areaUnderROC)