from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.sql import SparkSession, Row
import pandas as pd
import numpy as np




if __name__=='__main__':

    spark = SparkSession.builder \
        .appName('Mushroom') \
        .getOrCreate()

    mushrooms = spark.read.csv('kaggle-mushroom/input/mushrooms.csv')
    print mushrooms.dtypes