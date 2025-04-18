#=====================================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: Finds the most popular trajectories from extracted flight plans
#=====================================================================================================

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        prog='Flight Records Analyzer',
        description='Reads flight records (Sherlock headers) from the separated records .csv for analysis'
    )
    parser.add_argument('input_path', help='The input flight records data in .csv format')

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f'Provided input path: `{args.input_path}` does not exist')
        exit(1)
    
    # Read the records from the data file
    df = pd.read_csv(args.input_path)

    print(df.columns)

    # Filter where origin and destination both start with 'K'
    df_filtered = df[
        df['origin'].astype(str).str.startswith('K') &
        df['destination'].astype(str).str.startswith('K')
    ]

    # Count and sort route frequencies
    route_counts = (
        df_filtered
        .groupby(['origin', 'destination'])
        .size()
        .reset_index(name='count')
        .sort_values(by='count', ascending=False)
    )

    print(route_counts.head(10))

    # PHL to MCO


if __name__ == '__main__':
    main()