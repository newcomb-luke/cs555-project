import argparse
import os
import sys

# This will allow us to access sherlock-reader
sys.path.append('../')
from sherlock_reader import read_flights


def main():
    parser = argparse.ArgumentParser(
        prog='Flight Records Extractor',
        description='Reads flight records (Sherlock headers) from the NASA Sherlock data format for analysis'
    )
    parser.add_argument('input_path', help='The input Sherlock IFF data in .csv format')
    parser.add_argument('output_path', help='The path in which to output the header records for each flight in .csv format')

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f'Provided input path: `{args.input_path}` does not exist')
        exit(1)
    
    with open(args.output_path, 'w') as f:
        csv_header = 'time,flight_key,callsign,aircraft_type,origin,destination,ops_type,estimated_origin,estimated_dest\n'
        f.write(csv_header)

        # Read the flights from the Sherlock file
        with read_flights(args.input_path) as flights:
            for flight in flights:
                header = flight.header


                # Filter out bad data points
                if '?' == header.dest or 'unassigned' == header.dest or \
                   '?' == header.origin or 'unassigned' == header.origin:
                    print(f'Found flight {header.flt_key}...skipped')
                    continue

                print(f'Found flight {header.flt_key}...valid')

                csv_entry = f'{header.rec_time},{header.flt_key},{header.acid},{header.ac_type},{header.origin},{header.dest},{header.ops_type},{header.est_origin},{header.est_dest}\n'
                f.write(csv_entry)
    
    print('Done!')

if __name__ == '__main__':
    main()