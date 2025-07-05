import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score

def get_seed():
    # Set the seed for reproducibility, you can change this to any number
    return 73

def load_data_types(basepath: Path):
    # Some of the data types are going to be objects, we need to change them based on the data_type_dict json file. 
    # For the source and target entity do not include thier primary keys (costumer_id and product_id), beacuase
    # they will not be used as input features. They should be in a seperate variable
    with open(basepath / "data_type_dict.json", 'r') as file:
        dataTypeDic = json.load(file)

    src_type = dataTypeDic['src_entity']
    tar_type = dataTypeDic['tar_entity']

    src_entity = list(dataTypeDic['pkeys'].keys())[0]
    tar_entity = list(dataTypeDic['pkeys'].keys())[1]

    return src_type, tar_type, src_entity, tar_entity

def load_data(basepath: Path, src_type, tar_type, train=True):
    # Since we want to predict the list of product IDs a customer (that is identify with a customer ID) will buy, the source
    # entity is the customer and the target entity is the product. The reationship between them is the transactions table. 
    # Load the source and target entity tables, and force the correct data types
    source_entity_table = pd.read_csv(basepath / "customers.csv").astype(src_type)
    target_entity_table = pd.read_csv(basepath / "products.csv").astype(tar_type)

    data = []

    if train:
        # Load the data for training, validation
        train_data = pd.read_csv(basepath / "train_data.csv")
        val_data = pd.read_csv(basepath / "validation_data.csv")  

        # Replace spaces with comas in the list_product_id column so we can have a list of product_ids
        train_data['list_product_id'] = train_data['list_product_id'].apply(lambda text: text.replace(' ', ', '))
        val_data['list_product_id'] = val_data['list_product_id'].apply(lambda text: text.replace(' ', ', '))

        # Rename the columns of the train and valid data to match the target entity 
        train_data = train_data.rename(columns={'list_product_id': 'product_id'})
        val_data = val_data.rename(columns={'list_product_id': 'product_id'})

        data += [train_data, val_data]
    else:
        #Load the test data
        customer_id_df = pd.read_csv(basepath / "to_predict.csv")
        data += [customer_id_df]

    return data + [source_entity_table, target_entity_table]

def update_data(source_entity_table,target_entity_table,new_features, src_entity, src_type:dict, tar_entity, tar_type:dict):
    # Update the source and target entity tables with the new features
    source_entity_table = source_entity_table.merge(new_features[0], how="left", on=src_entity)
    target_entity_table = target_entity_table.merge(new_features[1], how="left", on=tar_entity)
    # Update the data types of the source and target entity tables with the new features
    src_type.update(new_features[2])
    tar_type.update(new_features[3])
    # Combine all data types
    all_types = src_type
    all_types.update(tar_type)

    return source_entity_table, target_entity_table, all_types


def negative_sampling(df: pd.DataFrame, tar_entity: str, tar_entity_df: pd.DataFrame, target_col_name: str) -> pd.DataFrame:
    # Create a negative sampling df, containing source and target entities pairs,
    # such that there are no links between them.
    negative_sample_df_columns = list(df.columns)
    negative_sample_df_columns.remove(tar_entity)
    negative_samples_df = df[negative_sample_df_columns]
    negative_samples_df[tar_entity] = np.random.choice(
        tar_entity_df[tar_entity], size=len(negative_samples_df)
    )
    negative_samples_df[target_col_name] = 0

    return negative_samples_df

# Change this function to include your new features and make your data for training is more complex
def feature_engineering(basepath: Path, src_entity: str, tar_entity: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    #Perform feature engineering on transaction data to extract customer and product features. You can create new features based on the existing ones, 
    # they dont need to be the same as the ones in the example below.
    
    # Load transaction data
    transaction_data = pd.read_csv(basepath / "transactions.csv")
    # Ensure transaction date is in datetime format
    transaction_data['time_date'] = pd.to_datetime(transaction_data['time_date'])

    # Create customer-related features
    customer_new_features = transaction_data.groupby(src_entity).agg({
        # Recency, first purchase date
        'time_date': ['max', 'min'],
        # Purchase frequency  
        src_entity: 'count',
        # Monetary value           
        'price': 'sum'                
    }).reset_index()
    # Flatten multi-level columns
    customer_new_features.columns = [src_entity, 'last_purchase_date', 'first_purchase_date', 'total_purchases', 'total_spent']
    # Create a dictionary to specify the data types of the new features
    customer_new_features_type = {'last_purchase_date': 'category', 'first_purchase_date': 'category', 'total_purchases': 'Int64', 'total_spent': 'float64'}
    customer_new_features.astype(customer_new_features_type)

    # Create product-related features
    product_new_features = transaction_data.groupby(tar_entity).agg({
        # Average price, price variation
        'price': ['mean', 'std'],
        # Product popularity  
        tar_entity: 'count'         
    }).reset_index()
    # Flatten multi-level columns
    product_new_features.columns = [tar_entity, 'average_price', 'price_std', 'popularity']
    # Create a dictionary to specify the data types of the new features
    product_new_features_type = {'average_price': 'float64', 'price_std': 'float64', 'popularity': 'Int64'}
    product_new_features.astype(product_new_features_type)

    return customer_new_features, product_new_features, customer_new_features_type, product_new_features_type

def get_split_train_val(dfs, target_col_name: str):
    # Split the features (X) and target (y) from train and validation and drop the target column from the features
    X_train = dfs["train"].drop(columns=[target_col_name])
    y_train = dfs["train"][target_col_name]

    X_val = dfs["val"].drop(columns=[target_col_name])
    y_val = dfs["val"][target_col_name]

    return X_train, y_train, X_val, y_val

def evaluate(xgb_model, X, y, typ: str):
    # Evaluate the model
    # Predict for the train and validation data
    y_pred = xgb_model.predict_proba(X)

    # Create a mask to convert the probabilities to binary values
    mask = (y_pred[:,1] >= 0.5).astype(int)

    # Calculate model metrics
    accuracy = accuracy_score(y, mask)
    precision = precision_score(y, mask)
    recall = recall_score(y, mask)
    map_T = np.mean(average_precision_score(y, mask))

    # Print the results
    print(f"{typ} Accuracy: {accuracy}")
    print(f"{typ} Presicion: {precision}")
    print(f"{typ} Recall: {recall}")
    print(f"{typ} Mean Average Precision: {map_T}\n")

def create_submission_file(basepath: Path):
    filepath = basepath / 'predictions.txt' 
    output_path = basepath / 'submission_file.csv'

    # Read the .txt file
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Initialize an empty list to store processed data
    data = []

    # Iterate through each line in the file, skipping the header if present
    for line in lines[1:]:
        # Extract the customer_id and the list of product_id from each line
        customer_id, product_list = line.strip().split(",", 1)
        # Use regular expressions to extract the product IDs (keeping the original list structure)
        product_ids = re.findall(r'\d+', product_list)
        # Join product IDs as a single space-separated string and store the result
        data.append([int(customer_id), '[' + ' '.join(product_ids) + ']'])  # maintain bracket format

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['customer_id', 'list_product_id'])

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False) 
