python3 PySparkMLFit.py --data_path=session-stat.parquet --model_path=spark_ml_model
python3 PySparkMLPredict.py --data_path=test.parquet --model_path=spark_ml_model --result_path=result
