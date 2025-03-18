from loader import load_token_data
from processor import remove_price_anomalies
import pandas as pd
import numpy as np
import math


# Given list of transactions

# 


def get_token_price_changes(token_data):
    return token_data["token_price"].pct_change() 

def get_trade_size_ratio(token_data):
    # How much they bought / sold of the initial supply
    if token_data["pool_spl_after"] > token_data["pool_spl_before"]:  # Sell
        return abs(token_data["pool_spl_after"] - token_data["pool_spl_before"]) / token_data["pool_spl_after"]
    elif token_data["pool_spl_after"] < token_data["pool_spl_before"]:  # Buy
        return abs(token_data["pool_spl_after"] - token_data["pool_spl_before"]) / token_data["pool_spl_before"]
    
def get_trade_liquidity_ratio(token_data):
    return token_data["pool_sol_before"] / token_data["pool_spl_before"]

def get_token_feature_vectors(token_address):
    # Load data
    df = load_token_data(token_address)
        
    # Clean data for price features
    cleaned_df = remove_price_anomalies(df)
    
    features = pd.DataFrame({
        "price_chnange_pct": None,
        "trade_size_ratio": None,
        "liquidity_ratio": None,
        "time": None

    })
    
    return features

def save_feature_vectors(token_addresses, output_file="../../data/Jan_25_Token_Features.csv"):
    all_features = []
    skipped_tokens = []
    
    for token in token_addresses:
        feature_vectors = get_token_feature_vectors(token)


if __name__ == "__main__":
    token_address = "XfNwSpXywvisWz5DEaW3STHh4YDAdqqexx1o9Bzpump"
    # Load data
    df = load_token_data(token_address)
        
    # Clean data for price features
    cleaned_df = remove_price_anomalies(df)

    price_change_pct = get_token_price_changes(cleaned_df)
    
