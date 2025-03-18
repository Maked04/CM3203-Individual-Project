# Entry point to test data methods
import loader
import processor
from cluster_feature_extraction import save_feature_vectors
from src.visualisation.price_plotter import plot_token_lifecycle, plot_token_price
import random


def main():
    token_address = random.choice(loader.list_available_tokens())
    #token_address = "6Wa7UyQHAAGQDjeXtT895cfoTRgWJVPAJxEDT1zJpump"
    df = loader.load_token_data(token_address)

    launch_df, peak_df, decline_df, full_df = processor.segment_token_lifecycle(df)
    
    # Plot the lifecycle phases
    plot_token_lifecycle(full_df, launch_df, peak_df, decline_df, token_address)

    #tokens = loader.list_available_tokens()
    #save_feature_vectors(tokens)
    

if __name__ == "__main__":
    main()