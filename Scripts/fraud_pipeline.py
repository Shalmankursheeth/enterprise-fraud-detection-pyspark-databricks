"""
Enterprise-Scale Fraud Detection Pipeline
-----------------------------------------
Author: Mohamed Shalman Kursheeth K

This script:
1. Loads credit-card transaction data
2. Preprocesses and engineers features with PySpark
3. Trains a LightGBM model
4. Evaluates metrics (AUC, Accuracy, Precision, Recall)
5. Saves the trained model for deployment
"""

import argparse
import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import array, udf
from pyspark.ml.linalg import Vectors, VectorUDT
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
from utils import log_message, feature_list

# ----------------------------
# Argument Parser
# ----------------------------
parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
parser.add_argument("--data-url", type=str, default="https://raw.githubusercontent.com/mayankyadav23/Credit-Card-Fraud-Detection/main/balanced_dataset.csv",
                    help="URL or path to CSV dataset")
parser.add_argument("--output-model", type=str, default="./models/fraud_model.pkl",
                    help="Path to save trained model")
args = parser.parse_args()

# ----------------------------
# Spark Session
# ----------------------------
spark = SparkSession.builder.appName("FraudDetectionPipeline").getOrCreate()
log_message(f"Spark session started with version {spark.version}")

# ----------------------------
# Load Data
# ----------------------------
log_message("Loading dataset...")
pdf = pd.read_csv(args.data_url)
pdf = pdf.drop(columns=['Time', 'Unnamed: 0'], errors='ignore')
df = spark.createDataFrame(pdf)
log_message(f"Loaded {df.count()} rows with {len(df.columns)} columns")

# ----------------------------
# Data Cleaning & Feature Engineering
# ----------------------------
label_col = "Class"

# Type casting
for c, t in df.dtypes:
    if c != label_col:
        df = df.withColumn(c, F.col(c).cast(DoubleType()))
df = df.withColumn(label_col, F.col(label_col).cast(IntegerType()))

# Derived features
v_cols = [c for c in df.columns if c.startswith("V")]
df = df.withColumn("Amount_abs", F.abs(F.col("Amount")))
df = df.withColumn("Amount_log1p", F.log1p(F.abs(F.col("Amount"))))
sq_sum = None
for c in v_cols:
    sq_sum = (sq_sum + (F.col(c) * F.col(c))) if sq_sum is not None else (F.col(c) * F.col(c))
df = df.withColumn("txn_magnitude", F.sqrt(sq_sum))

feature_cols = feature_list(v_cols)
log_message(f"Feature columns: {feature_cols}")

# Vectorize
def arr_to_vector(arr):
    return Vectors.dense([0.0 if x is None else float(x) for x in arr])

arr_to_vector_udf = udf(arr_to_vector, VectorUDT())
df = df.withColumn("features_array", array(*[F.col(c).cast(DoubleType()) for c in feature_cols]))
df = df.withColumn("features", arr_to_vector_udf(F.col("features_array"))).select("features", label_col)

# ----------------------------
# Train-Test Split
# ----------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_pd = train_df.toPandas()
test_pd = test_df.toPandas()

X_train = np.stack(train_pd["features"].apply(lambda v: v.toArray()))
y_train = train_pd[label_col].values
X_test = np.stack(test_pd["features"].apply(lambda v: v.toArray()))
y_test = test_pd[label_col].values

# ----------------------------
# Train LightGBM
# ----------------------------
log_message("Training LightGBM model...")
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
log_message(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}")
log_message("Classification Report:\n" + classification_report(y_test, y_pred))
log_message("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

# ----------------------------
# Save Model
# ----------------------------
os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
joblib.dump(model, args.output_model)
log_message(f"Model saved at {args.output_model}")
