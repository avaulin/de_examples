import operator
import argparse

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import GBTClassifier

MODEL_PATH = 'spark_ml_model'
LABEL_COL = 'is_bot'


def process(spark, data_path, model_path):
    """
    Основной процесс задачи.

    :param spark: SparkSession
    :param data_path: путь до датасета
    :param model_path: путь сохранения обученной модели
    """

    train_df = spark.read.parquet(data_path)

    assembler = vector_assembler()
    evaluator = build_evaluator()

    user_type_index = StringIndexer(inputCol='user_type', outputCol="user_type_index")
    platform_index = StringIndexer(inputCol='platform', outputCol="platform_index")
    platform_ohe = OneHotEncoder().setInputCol('platform_index').setOutputCol('platform_vec')

    gbt = GBTClassifier(labelCol=LABEL_COL, featuresCol='features')
    pipe = Pipeline(
        stages=[user_type_index, platform_index, platform_ohe, assembler, gbt]
    )

    cv = CrossValidator(estimator=pipe, estimatorParamMaps=model_params(gbt), evaluator=evaluator)

    models = cv.fit(train_df)
    best = models.bestModel
    preds = best.transform(train_df)
    f1 = evaluator.evaluate(preds)
    print(f1)
    best.write().overwrite().save(MODEL_PATH)
    train_df.show(10)


def main(data_path, model_path):
    spark = _spark_session()
    process(spark, data_path, model_path)


def model_params(model):
    return ParamGridBuilder() \
        .addGrid(model.maxDepth, [5, 10]) \
        .addGrid(model.maxBins, [2, 5, 10]) \
        .build()


def vector_assembler() -> VectorAssembler:
    features = [
        'user_type_index',
        'duration',
        'platform_vec',
        'item_info_events',
        'select_item_events',
        'make_order_events',
        'events_per_min',
    ]
    return VectorAssembler(
        inputCols=features,
        outputCol='features'
    )


def build_evaluator() -> MulticlassClassificationEvaluator:
    return MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol='prediction',
        metricName='f1'
    )


def _spark_session():
    """
    Создание SparkSession.

    :return: SparkSession
    """
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='session-stat.parquet', help='Please set datasets path.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Please set model path.')
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    main(data_path, model_path)
