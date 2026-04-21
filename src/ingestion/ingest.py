from pyspark.sql import functions as F

def ingest_raw_to_delta(spark, input_path, output_table):
    df_raw = spark.read.parquet(input_path)
    
    df_clean = (df_raw
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("fare_amount") > 0)
        .filter(F.col("tpep_pickup_datetime").isNotNull())
        .filter(F.col("tpep_dropoff_datetime").isNotNull())
        .withColumn("trip_duration_seconds",
            F.unix_timestamp("tpep_dropoff_datetime") -
            F.unix_timestamp("tpep_pickup_datetime"))
        .filter(F.col("trip_duration_seconds").between(60, 10800))
        .withColumn("pickup_date", F.to_date("tpep_pickup_datetime"))
    )
    
    df_clean.write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("pickup_date") \
        .saveAsTable(output_table)
    
    return df_clean.count()