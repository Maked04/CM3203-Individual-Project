# FOR CLEANING AND PREPROCESSING

import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler


"""
Raydium schema 

        tx.tx_sig,             # 1st value 
        tx.block_time,         # 2nd value 
        tx.slot,               # 3rd value 
        tx.fee,                # 4th value 
        tx.token_price,        # 5th value 
        tx.token_address,      # 6th value 
        tx.is_creator,         # 7th value 
        tx.signer,             # 8th value 
        tx.pool_spl_before,      # 9th value 
        tx.pool_spl_after,       # 10th value 
        tx.pool_wsol_before,      # 11th value 
        tx.pool_wsol_after,       # 12th value 
        tx.signer_spl_before,  # 13th value 
        tx.signer_spl_after,   # 14th value 
        tx.signer_sol_change  # 15th value 
  

PumpFun schema 

        tx.tx_sig,             # 1st value 
        tx.block_time,         # 2nd value 
        tx.slot,               # 3rd value 
        tx.fee,                # 4th value 
        tx.token_price,        # 5th value 
        tx.token_address,      # 6th value 
        tx.is_creator,         # 7th value 
        tx.signer,             # 8th value 
        tx.bc_spl_before,      # 9th value 
        tx.bc_spl_after,       # 10th value 
        tx.bc_sol_before,      # 11th value 
        tx.bc_sol_after,       # 12th value 
        tx.signer_spl_before,  # 13th value 
        tx.signer_spl_after,   # 14th value 
        tx.signer_sol_before,  # 15th value 
        tx.signer_sol_after    # 16th value 


Standard schema

    tx_sig: str
    block_time: int
    slot: int
    fee: int
    token_price: Decimal
    token_address: str
    is_creator: bool
    signer: str
    pool_spl_before: Decimal
    pool_spl_after: Decimal
    pool_sol_before: Decimal
    pool_sol_after: Decimal
    signer_spl_before: Decimal
    signer_spl_after: Decimal
    signer_sol_change: Decimal
"""

def get_inital_liquidity(trades):
    first_trade = trades[0]
    if first_trade["is_creator"]:
        return first_trade["pool_sol_after"]
    else:
        return None
    
def get_active_lifetime(trades):
    return trades[-1]["block_time"] - trades[0]["block_time"]

def get_unique_trader_count(trades):
    return len(set(trade["token_address"] for trade in trades))

def get_average_trading_size(trades):
    '''Returns average trade size in Sol'''
    return sum(trade["signer_sol_change"] for trade in trades) / len(trades)

def get_max_market_cap(trades):
    return max((trade["pool_sol_after"] for trade in trades))

def trade_frequency(trades):
    get_active_lifetime(trades) / len(trades)


def remove_anomalies_mad(trades, threshold=3):
    prices = [trade["token_price"] for trade in trades]

    if len(prices) == 0:
        return []
    median = np.median(prices)
    mad = np.median([abs(price - median) for price in prices])
    
    if mad == 0:
        return prices
        
    return [
        trades for trade in trades 
        if abs(trade["token_price"] - median) / mad < threshold
    ]


def get_token_feature_vector(trades: List[Dict]) -> np.ndarray:
    """
    Calculate feature vector for token trading activity, formatted for sklearn k-means.
    Returns a 1D numpy array of features.
    
    Args:
        trades: List of standardized trade dictionaries
        
    Returns:
        np.ndarray: 1D array of features [
            initial_liquidity,
            active_lifetime,
            unique_traders,
            avg_trade_size,
            max_market_cap,
            trade_frequency,
            price_volatility,
            total_volume,
            price_trend,
            liquidity_concentration,
            creator_activity
        ]
    """
    # Remove anomalies first
    cleaned_trades = remove_anomalies_mad(trades)
    
    if not cleaned_trades:
        return np.zeros(11)  # Return zero vector if no valid trades
    
    # Basic features using existing functions
    initial_liq = float(get_inital_liquidity(cleaned_trades) or 0)  # Convert None to 0
    lifetime = float(get_active_lifetime(cleaned_trades))
    unique_traders = float(get_unique_trader_count(cleaned_trades))
    avg_trade_size = float(get_average_trading_size(cleaned_trades))
    max_cap = float(get_max_market_cap(cleaned_trades))
    avg_frequency = lifetime / len(cleaned_trades) if len(cleaned_trades) > 0 else 0
    
    # Additional features
    prices = [trade["token_price"] for trade in cleaned_trades]
    price_volatility = float(np.std(prices)) if len(prices) > 0 else 0
    
    total_volume = float(abs(sum(trade["signer_sol_change"] for trade in cleaned_trades)))
    
    if len(cleaned_trades) >= 2:
        initial_price = cleaned_trades[0]["token_price"]
        final_price = cleaned_trades[-1]["token_price"]
        price_trend = (final_price / initial_price - 1) if initial_price != 0 else 0
    else:
        price_trend = 0
    
    liquidities = [trade["pool_sol_after"] for trade in cleaned_trades]
    avg_liquidity = np.mean(liquidities) if liquidities else 0
    liquidity_concentration = max_cap / avg_liquidity if avg_liquidity != 0 else 0
    
    creator_trades = sum(1 for trade in cleaned_trades if trade["is_creator"])
    creator_activity = creator_trades / len(cleaned_trades) if cleaned_trades else 0
    
    # Return as 1D numpy array
    return np.array([
        initial_liq,
        lifetime,
        unique_traders,
        avg_trade_size,
        max_cap,
        avg_frequency,
        price_volatility,
        total_volume,
        price_trend,
        liquidity_concentration,
        creator_activity
    ])

def get_feature_vectors_for_tokens(token_trades_list: List[List[Dict]], scale: bool = True) -> np.ndarray:
    """
    Convert multiple tokens' trades into a feature matrix suitable for k-means clustering.
    
    Args:
        token_trades_list: List of trade lists, where each inner list contains trades for one token
        scale: Whether to standardize features using StandardScaler
        
    Returns:
        np.ndarray: 2D array of shape (n_tokens, n_features)
    """
    # Get feature vectors for all tokens
    feature_vectors = [get_token_feature_vector(trades) for trades in token_trades_list]
    
    # Stack into 2D array
    X = np.vstack(feature_vectors)
    
    # Scale features if requested
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X