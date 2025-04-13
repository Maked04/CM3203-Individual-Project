import pandas as pd
import numpy as np
import os


def generate_cumulative_wallet_metrics(wallet_csv):
    # Load your CSV
    df = pd.read_csv(wallet_csv)

    metric_df = pd.DataFrame()

    df["volume_this_tx"] = np.abs(df["bc_sol_after"] - df["bc_sol_before"])
    df["sol_change"] = (df["bc_sol_after"] - df["bc_sol_before"])

    # Add block time to metric df
    metric_df["block_time"] = df["block_time"]

    # Cumulative volume PRIOR to each tx (shifted)
    metric_df["volume_prior"] = df["volume_this_tx"].cumsum().shift(1).fillna(0)

    # 📈 Number of prior trades (simple count)
    metric_df["trade_count_prior"] = df.reset_index().index.to_series().shift(1).fillna(0)

    # 📊 Deviation metric — current trade vs average prior trade size
    avg_trade_prior = df["volume_this_tx"].expanding().mean().shift(1)
    metric_df["trade_size_deviation"] = (df["volume_this_tx"] - avg_trade_prior).fillna(0)

    # SOME SORT OF SUCCESS METRIC
    metric_df["rough_pnl"]  = df["sol_change"].cumsum().shift(1).fillna(0)

        

    return metric_df


def process_wallets(input_directory, output_directory):
    files = os.listdir(input_directory)
    files = files[1: 3]
    os.makedirs(output_directory, exist_ok=True)

    for file in files:  # Per wallet
        metric_df = generate_cumulative_wallet_metrics(wallet_csv=os.path.join(input_directory, file))
        metric_df.to_csv(os.path.join(output_directory, file), index=False)


if __name__ == "__main__":
    input_directory = "../../data/wallet_data"
    output_directory = "../../data/wallet_metrics"
    process_wallets(input_directory, output_directory)

