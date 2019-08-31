'''
Created on Jun 17, 2019
@author: upagupta
Finding Jaccard Similarity between two different words
'''

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover

import pandas as pd

def uniqueWords(inputWordList):
    unique_list = []
    for x in inputWordList:
        if x not in unique_list:
            unique_list.append(x)
    return (unique_list)

def count_pairs(line):
    from itertools import combinations 
    all_pairs = combinations(line, 2)
    all_pairs_sorted = (sorted(x) for x in all_pairs)
    return((all_pairs_sorted))

def pairCounts(tuples):
    total_count = 0
    res_tup = []
    for t in tuples:
        res_tup.append(t[0])
        total_count =  total_count+t[1]
    return(tuple(res_tup), total_count)

def JaccardSimilarity(inputs):
    pairs = inputs[0]
    values = inputs[1]
    intersection_value = values[0]
    union_value = values[1]-values[0]
    similarity = (float(intersection_value)/float(union_value))
    return(pairs,similarity)

sparkConf = SparkConf().setAppName("JaccardSim").setMaster("local")
sc = SparkContext(conf = sparkConf)
sqlContext = SQLContext(sc)


if __name__ == '__main__':
    
    data = pd.read_csv('movie_data.csv')
    dataPd = sqlContext.createDataFrame(data)
    
    tokenizer = Tokenizer(inputCol="review", outputCol="words")

    dataPd2 = tokenizer.transform(dataPd)
    
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    dataPd3 = remover.transform(dataPd2)

    dataReviewsTokenized = dataPd3.select('filtered').rdd
    
    dataReviewsTokenizedSet = dataReviewsTokenized.flatMap(list).map(uniqueWords)
    
    distinctWords = dataReviewsTokenizedSet.flatMap(lambda x : list(x)).distinct()
    dataTokenized = dataReviewsTokenizedSet
    
    WordCounts = dataTokenized.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a+b)

    AllPossiblePairs = distinctWords.cartesian(distinctWords)
    AllPossiblePairs = AllPossiblePairs.map(lambda x: tuple(sorted(x))).distinct()
    
    PairCounts = dataTokenized.flatMap(count_pairs).map(tuple)
    DistinctPairCounts = PairCounts.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a+b)
    
    AllPossiblePairs = AllPossiblePairs.map(lambda x: (x,0))
    
    AllPossiblePairCounts = AllPossiblePairs.union(DistinctPairCounts)
    AllPossiblePairCounts = AllPossiblePairCounts.reduceByKey(lambda a, b: a+b)
    
    WordCountPairs =  WordCounts.cartesian(WordCounts).map(lambda x: tuple(sorted(x))).distinct()
    
    WordPairAggregateCount = WordCountPairs.map(pairCounts)
    
    WordPairs_all_counts = AllPossiblePairCounts.fullOuterJoin(WordPairAggregateCount)
    
    JaccardSim = WordPairs_all_counts.map(JaccardSimilarity)
    Results = JaccardSim.toPandas()
    Results.to_csv('jaccard_similarity.csv', index=False)