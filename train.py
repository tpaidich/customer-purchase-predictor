import os
import random
import numpy as np
import pandas as pd
from typing import Dict
from ast import literal_eval
from pathlib import Path

import data_utils as du
from xgboost import XGBClassifier

BASEPATH = Path(os.path.dirname(__file__))
SEED = du.get_seed()

def train():
    # Set the seed for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)

    # Load the data types and data from the csv files
    src_type, tar_type, src_entity, tar_entity = du.load_data_types(BASEPATH)
    train_data, val_data, source_entity_table, target_entity_table = du.load_data(BASEPATH, src_type, tar_type)

    # Prepare train/val dataset for XGBoost model training.
    # Add the new features to the source and target entity tables
    new_features = du.feature_engineering(BASEPATH, src_entity, tar_entity)
    source_entity_table, target_entity_table, all_types = du.update_data(
        source_entity_table,
        target_entity_table,
        new_features,
        src_entity,
        src_type,
        tar_entity,
        tar_type
    )

    # Define the target column name
    target_col_name = "link_pred_baseline_target"
    # Create a dictionary to store the dataframes
    dfs: Dict[str, pd.DataFrame] = {}

    for split, table in [("train", train_data), ("val", val_data)]:
        # Left join train table and entity table
        df = table.merge(source_entity_table, how="left", left_on=src_entity, right_on=src_entity)
        # Transform the mapping between one source entity with a list of target entities to source entity, target entity pairs
        df["product_id"] = df["product_id"].apply(literal_eval)
        df = df.explode("product_id")
        # Add a target col indicating there is a link between source and target
        df[target_col_name] = 1

        # Perform Negative Sampling (For each source entity use their corresponding target entities as positive labels. The same number 
        # of random target entities, that do not correspond to the source entity, are sampled as negative labels.)
        negative_samples_df = du.negative_sampling(df, tar_entity, target_entity_table, target_col_name)
        # Constructing a dataframe containing the same number of positive and negative links
        df = pd.concat([df, negative_samples_df], ignore_index=True)
        # Left join train table (which has been previously joined with source entity table) with the target entity table
        df = pd.merge(df, target_entity_table, how="left", left_on=tar_entity, right_on=tar_entity)
        
        # Change the data types to the correct ones and drop the unnecessary columns
        df['time_date'] = df['time_date'].astype('category', errors='ignore')
        df = df.astype(all_types)

        # Drop the customer_id and product_id columns, since they are not needed for training
        df.drop(columns=[src_entity, tar_entity], inplace=True)

        # Store the dataframe in the dictionary
        dfs[split] = df

    # Split the features (X) and target (y) from train and validation
    X_train, y_train, X_val, y_val = du.get_split_train_val(dfs, target_col_name)
    
    # Create model instance (do hyperparameters seach to find the best model)
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.9,
        objective='binary:logistic',
        eval_metric='auc',
        enable_categorical=True,
        device='cuda',
        predictor='gpu_predictor'
    )

    # Fit model
    xgb_model.fit(X_train, y_train)

    # Save model
    booster = xgb_model.get_booster()
    booster.set_param({"device": "cuda:0"})
    booster.save_model('XGBoost_model.json')

    # Evaluate the model
    du.evaluate(xgb_model, X_train, y_train, "Training")
    du.evaluate(xgb_model, X_val, y_val, "Validation")

if __name__ == '__main__':
    train()