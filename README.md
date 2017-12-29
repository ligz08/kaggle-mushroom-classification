# Kaggle Mushroom Classification Challenge
[Kaggle Mushroom Classification Challenge](https://www.kaggle.com/uciml/mushroom-classification) 
 with Spark MLLib.

## Set up environment variable for Spark 2
When both Spark 1 and 2 are installed on a machine, it by default uses Spark 1,
and you need explicitly tell the machine to use Spark 2 by running this command before using `pyspark` or `spark-submit`:
```bash
$ export SPARK_MAJOR_VERSION=2 
```

## Import modules
I used the following modules from `pyspark` for this task
```python
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
```

## Set up `SparkContext` and `SparkSession`
```python
sc = SparkContext()

spark = SparkSession.builder \
    .appName('Mushroom') \
    .getOrCreate()
```

## Read input .csv file
Before loading the csv file into Spark, first put it in HDFS.
```bash
$ hdfs dfs -put mushrooms.csv kaggle-mushroom-classification/input/
```
Then load it using Spark's `read.csv()` function:
```python
mushrooms = spark.read.csv('kaggle-mushroom/input/mushrooms.csv', header=True)
```
Check schema to make sure that works:
```text
>>> mushrooms.printSchema()
root
 |-- class: string (nullable = true)
 |-- cap-shape: string (nullable = true)
 |-- cap-surface: string (nullable = true)
 |-- cap-color: string (nullable = true)
 |-- bruises: string (nullable = true)
 |-- odor: string (nullable = true)
 |-- gill-attachment: string (nullable = true)
 |-- gill-spacing: string (nullable = true)
 |-- gill-size: string (nullable = true)
 |-- gill-color: string (nullable = true)
 |-- stalk-shape: string (nullable = true)
 |-- stalk-root: string (nullable = true)
 |-- stalk-surface-above-ring: string (nullable = true)
 |-- stalk-surface-below-ring: string (nullable = true)
 |-- stalk-color-above-ring: string (nullable = true)
 |-- stalk-color-below-ring: string (nullable = true)
 |-- veil-type: string (nullable = true)
 |-- veil-color: string (nullable = true)
 |-- ring-number: string (nullable = true)
 |-- ring-type: string (nullable = true)
 |-- spore-print-color: string (nullable = true)
 |-- population: string (nullable = true)
 |-- habitat: string (nullable = true)
```

## Preprocess feature columns
One-hot encode the categorical feature columns. In our case all features are categorial.
```python
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
```
The `str_indexers` are responsible for converting string type values (like `a` `b` `c`) in our columns to numbers (like `0` `1` `2`).  
The `onehot_encoders` are responsible for converting numeric category labels to one-hot encoding.  
`label_indexer` converts the target labels (`e` and `p`) to `0` and `1`. 
By default the `StringIndexer` object gives smaller labels to more frequent classes. In our case, `e` appears more often then `p`. This can be checked by:
```text
>>> mushrooms.groupby('class').count().show()
+-----+-----+
|class|count|
+-----+-----+
|    e| 4208|
|    p| 3916|
+-----+-----+
```

## Train-test split
```python
mushrooms_train, mushrooms_val = mushrooms_trans.randomSplit([0.7, 0.3], seed=2017)
```

## Random forest model
```python
model = RandomForestClassifier(labelCol='poisonous', featuresCol='features', numTrees=200) \
        .fit(mushrooms_train)
```

## Predict and evaluate model
```python
pred = model.transform(mushrooms_val)

results = pred.select(['probability', 'prediction', 'poisonous'])
# Select the columns relevant for evaluation
# `results` looks like this:
# +--------------------+----------+---------+
# |         probability|prediction|poisonous|
# +--------------------+----------+---------+
# |[0.97024593961675...|       0.0|      0.0|
# |[0.96303265951929...|       0.0|      0.0|
# |[0.95909221894651...|       0.0|      0.0|
# |[0.95958294573868...|       0.0|      0.0|
# |[0.95580449199223...|       0.0|      0.0|
# +--------------------+----------+---------+

results_collect = results.collect()
# After .collect(), `results_collect` become a list of Row objects

correct = results.withColumn('correct', (results.prediction==results.poisonous).cast('integer')).select('correct')

accuracy = correct.agg({'correct':'mean'}).collect()[0][0]

print('Test accuracy:', accuracy)
```
We finally get an accuracy of 0.9925 on the test set.

## Complete Python script
Please see [script/mushroomForest.py](script/mushroomForest.py)

## References
- Lucas Allen (2017), [*A New Introduction to Spark 2.1 Dataframes with Python and MLlib*](http://www.techpoweredmath.com/introduction-spark-2-1-dataframes-python-mllib/#.WkWRxlQ-fOR)
- Weimin Wang (2016), *PySpark case - using Random Forest for binary classification problem - PyDataSG.* [Presentation video on YouTube](https://www.youtube.com/watch?v=CdHuLGuU2c4).
- Ankit Gupta (2016), [*Complete Guide on DataFrame Operations in PySpark*](https://www.analyticsvidhya.com/blog/2016/10/spark-dataframe-and-operations/)
