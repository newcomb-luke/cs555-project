import json
import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser(
        'Custom Data Splitter',
        description='Splits a collection of .json files into train, test, and validation based on the provided ratios'
    )
    parser.add_argument('input_directory', help='The input directory containing the .json files')
    parser.add_argument('output_directory', help='The directory to output train.json, dev.json, and test.json')
    parser.add_argument('--train', default=0.8, type=float, help='The percentage of the total file which will be used for training')
    parser.add_argument('--validation', default=0.1, type=float, help='The percentage of the total file which will be used for validation. Testing is the left over')
    parser.add_argument('--shuffle', action="store_true", help='If the data should be shuffled before separating')
    
    args = parser.parse_args()

    if not os.path.exists(args.input_directory):
        print(f'Error: Input directory `{args.input_directory}` does not exist')
        exit(1)
    
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    
    if (args.train + args.validation) >= 1.0:
        print(f'Error: Train + Validation percentages are greater than or equal to 1')
        exit(2)
    
    test = 1.0 - (args.train + args.validation)

    print('Splitting:')
    print(f'  Train: {args.train * 100.0:.2f}%')
    print(f'  Test: {test * 100.0:.2f}%')
    print(f'  Validation: {args.validation * 100.0:.2f}%\n')

    file_paths = files_in_directory(args.input_directory)

    print(f'Found {len(file_paths)} files')

    json_data = load_files(file_paths)

    entries = join_files(json_data)

    if args.shuffle:
        print('Shuffling entries')
        random.shuffle(entries)

    print('Counting number of total entries...')

    total_entries = len(entries)

    print(f'Found {total_entries} total entries.\n')

    train_entries = int(args.train * total_entries)
    validation_entries = int(args.validation * total_entries)
    test_entries = total_entries - (train_entries + validation_entries)

    print('Entries per category:')
    print(f'  Train: {train_entries}')
    print(f'  Test: {test_entries}')
    print(f'  Validation: {validation_entries}\n')

    train_path = os.path.join(args.output_directory, 'train.json')
    validation_path = os.path.join(args.output_directory, 'dev.json')
    test_path = os.path.join(args.output_directory, 'test.json')

    train_data, val_data, test_data = split_data(entries, train_entries, test_entries)

    # Training
    print('Writing training file')
    write_to_file(train_data, train_path)

    # Validation
    print('Writing validation file')
    write_to_file(val_data, validation_path)

    # Test
    print('Writing test file')
    write_to_file(test_data, test_path)


def load_files(json_paths: list[str]) -> list[dict]:
    dicts = []

    for path in json_paths:
        with open(path, 'r') as f:
            json_data = json.load(f)
            dicts.append(json_data)
    
    return dicts


def files_in_directory(dir: str) -> list[str]:
    paths = []

    for entry in os.scandir(dir):
        if entry.name.endswith('json'):
            paths.append(entry.path)
    
    return paths


def join_files(json_data: list[dict]) -> list[dict]:
    entries = []
    for file_data in json_data:
        entries.extend(file_data['e'])
    return entries


def split_data(data: list[dict], num_train: int, num_val: int):
    train = data[:num_train]

    # Will be split into test and validation
    rest = data[num_train:]

    val = rest[:num_val]
    test = rest[num_val:]

    return train, val, test


def write_to_file(entries, output_path: str):
    json_dict = {
        'e': entries
    }

    with open(output_path, 'w') as f:
        json.dump(json_dict, f)


if __name__ == "__main__":
    main()