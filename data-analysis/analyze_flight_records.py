import argparse
import os
import sys
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

    print(df['destination'].value_counts())
    print(df['origin'].value_counts())

    print(df['destination'].apply(lambda d: d.startswith('K')).value_counts())

    print('======================================')

    print(df['estimated_dest'].value_counts())
    print(df['estimated_origin'].value_counts())

    print(df['ops_type'].value_counts())
    print(df['aircraft_type'].value_counts().nlargest(20))


if __name__ == '__main__':
    main()