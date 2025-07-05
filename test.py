import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
import data_utils as du

BASEPATH = Path(os.path.dirname(__file__))
MODEL_NAME = 'XGBoost_model.json'
SEED = du.get_seed()
NUM_CUSTOMERS = 1000

def __explode(customer_id_df, start_idx, num_customers, source_entity_table, src_entity, target_entity_table):
    # Select get the x customer_ids starting from start_idx
    df_test: pd.DataFrame = customer_id_df.iloc[start_idx:start_idx + num_customers][['customer_id']].reset_index(drop=True)

    # Left join df_chuch and source entity table (map the customer_ids from the test data to their corresponding features)
    df_test = df_test.merge(source_entity_table, how="left", left_on=src_entity, right_on=src_entity)
    # Left join test table that was previously joined with source entity table, with the target entity table in order to have 
    # all the prodcuts for all the customers (map each one of the customer_ids in the test table to all the product_ids 
    # in the target entity table)
    df_test = df_test.merge(target_entity_table, how='cross')

    # Add the time_date to each customer_id
    df_test['time_date'] = '2020-09-14'
    # Change the data type of the time_date to be categorical in order to pass it to the model
    df_test['time_date'] = df_test['time_date'].astype('category', errors='ignore')

    return df_test

def __order_and_cast(df_test, customers_features, products_features, all_types) -> pd.DataFrame:
    # Order the columns in the dataframe to match the order of the data used to train the model
    order_df = ['time_date', 'customer_id', 'product_id'] + list(customers_features.columns) + list(products_features.columns)
    df_test = df_test[order_df]
    # Change the data types to the correct ones
    df_test = df_test.astype(all_types)

    return df_test

# Function to get the top 12 products_ids for each customer_id
def __get_top_k(target_entity_table, probabilities, df, src_entity, tar_entity):
    customer_predictions = []
    # Get the number of products in the target entity table
    num_products = len(target_entity_table)
    idx = 0
    while idx < len(probabilities):
        # Select the current customer
        customer = df.loc[idx, src_entity]
        # Get the predicted probabilities of all products for the current customer
        prob_chunk = probabilities[idx:idx+num_products]

        # Get the indexes of the top 12 products (highest predicted probabilities)
        indexes = np.argsort(prob_chunk)[-12:]
        # Filter these to keep only probabilites above 0.5
        indexes = indexes[prob_chunk[indexes] >= 0.5] + idx

        prod_ids = df.loc[indexes, tar_entity].values.tolist()
        # Add the customer_id and its top 12 predictions to the list
        customer_predictions.append(f"{customer},{prod_ids}\n")
        # Update the index to correspond to the next customer
        idx += num_products
        
    return customer_predictions

def test():
    # Set the seed for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)

    # Open a file to write the predictions
    filepath = BASEPATH / 'predictions.txt'
    f = open(filepath, 'w')
    # Load model from file
    loaded_model = XGBClassifier()
    loaded_model.load_model(BASEPATH / MODEL_NAME)
    
    # Load the data types and data from the csv files
    src_type, tar_type, src_entity, tar_entity = du.load_data_types(BASEPATH)
    customer_ids, source_entity_table, target_entity_table = du.load_data(BASEPATH, src_type, tar_type, train=False)

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

    # Get the customer and product features
    customers_features = source_entity_table.drop(columns=[src_entity])
    products_features = target_entity_table.drop(columns=[tar_entity])

    # Write the header of the file
    f.write(f"{src_entity}, list_product_id\n")
    f.flush()

    # Initialize start_idx to know where to start selecting customer_ids from in each loop
    start_idx = 0
    # Loop through the number of customer_ids selected to get thier predictions
    tot_rows = len(customer_ids)
    while tot_rows > 0:
        num_customers = NUM_CUSTOMERS if tot_rows >= NUM_CUSTOMERS else tot_rows
        tot_rows -= NUM_CUSTOMERS

        # Create the test batch and add all necessary columns
        df_test = __explode(customer_ids, start_idx, num_customers, source_entity_table, src_entity, target_entity_table)
        
        # Force correct columns order and cast the right dtypes
        df_test = __order_and_cast(df_test, customers_features, products_features, all_types)

        # Make predictions
        predic_test = loaded_model.predict_proba(df_test.drop(columns=[src_entity, tar_entity]))
        probability_class_1 = predic_test[:, 1]

        # Get the top 12 products for each customer
        customer_predictions = __get_top_k(target_entity_table, probability_class_1, df_test, src_entity, tar_entity)

        # Write the customer_id and its top 12 predictions to the file
        f.writelines(customer_predictions)
        f.flush()
        
        # Delete all dataframes and list from memory
        del df_test
        
        # Update start_idx to correspond to the next x rows from the test customer_id dataframe
        start_idx += num_customers

    f.close()
    # Finalize the predictions file, making it ready for Kaggle
    du.create_submission_file(BASEPATH)

if __name__ == '__main__':
    test()