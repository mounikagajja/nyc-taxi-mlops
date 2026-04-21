from pyspark.sql import functions as F

def build_features(spark, input_table, output_table):
    df = spark.read.format("delta").table(input_table)

    df_features = (df
        .withColumn("hour_of_day", F.hour("tpep_pickup_datetime"))
        .withColumn("day_of_week", F.dayofweek("tpep_pickup_datetime"))
        .withColumn("is_weekend", F.when(
            F.dayofweek("tpep_pickup_datetime").isin([1, 7]), 1).otherwise(0))
        .withColumn("is_rush_hour", F.when(
            F.hour("tpep_pickup_datetime").isin([7,8,17,18,19]), 1).otherwise(0))
        .withColumn("is_night", F.when(
            (F.hour("tpep_pickup_datetime") >= 22) |
            (F.hour("tpep_pickup_datetime") <= 5), 1).otherwise(0))
        .withColumn("distance_km", F.col("trip_distance") * 1.60934)
    )

    feature_cols = [
        "trip_duration_seconds", "distance_km", "hour_of_day",
        "day_of_week", "is_weekend", "is_rush_hour", "is_night",
        "passenger_count", "PULocationID", "DOLocationID", "pickup_date"
    ]

    df_final = df_features.select(feature_cols).dropna()

    df_final.write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("pickup_date") \
        .saveAsTable(output_table)

    return df_final.count()