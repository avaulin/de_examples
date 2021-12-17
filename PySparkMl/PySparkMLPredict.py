import argparse

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

MODEL_PATH = 'spark_ml_model'


def main(data_path, model_path, result_path):
    """
    Применение сохраненной модели.

    :param data_path: путь к файлу с данными к которым нужно сделать предсказание.
    :param model_path: путь к сохраненой модели (Из задачи PySparkMLFit.py).
    :param result_path: путь куда нужно сохранить результаты предсказаний ([session_id, prediction]).
    """
    spark = _spark_session()
    test_df = spark.read.parquet(data_path)

    model = PipelineModel.load(model_path)
    preds = model.transform(test_df)
    preds.select(col('session_id'), col('prediction')).write.save(result_path)

    preds.show(10)


def _spark_session():
    """
    Создание SparkSession.

    :return: SparkSession
    """
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Please set model path.')
    parser.add_argument('--data_path', type=str, default='test.parquet', help='Please set datasets path.')
    parser.add_argument('--result_path', type=str, default='result', help='Please set result path.')
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    result_path = args.result_path
    main(data_path, model_path, result_path)
