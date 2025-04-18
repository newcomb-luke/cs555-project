# CS 555 Trajectory Prediction Project

Our ML project for CS 555

This project predicts commercial aircraft trajectories using a transformer-based sequence model. It processes raw flight surveillance data, constructs structured trajectory datasets, and trains a neural model to predict future motion conditioned on route information.
The implementation is modular and includes components for:
* Reading and parsing FAA airspace reference data (airports, fixes, airways, etc.)
* Parsing and validating raw flight data from NASA's Sherlock database
* Converting trajectories to fixed-length, normalized training sequences
* Training and evaluating a transformer model for trajectory prediction
* Plotting and visualizing trajectories and flight plans

## Project Structure

### faa_reader
Parses FAA reference CSV files. These files contain the up-to-date locations of all airports, fixes, navigational aids, and airways in the US airspace.

### sherlock_reader
Parses NASA Sherlock flight data in the IFF CSV format
* Supports header, flight plan, and tracking point records
* Resolves text-based flight plans into structured sequences of coordinates
* Used to validate and convert real flight data for use in the final dataset

### data-analysis/
Various files used to visualize and sort/rank the data in order to explore the properties of the dataset.

### data-processing/data_splitter.py
Splits collected JSON dataset files into train, dev, and test splits

### model
Implements the real training/evaluation/inference scripts for this project.
Contains:
* model.py
* train.py
* eval.py
* infer.py

## Dependencies

* torch
* numpy
* matplotlib
* pandas
* scipy

Install dependencies using:
```bash
pip install torch numpy matplotlib pandas scipy
```

Tested with Python 3.12

## Usage

### Step 1. FAA Data + Sherlock Data
* Download the FAA reference files from the [FAA website](https://www.faa.gov/air_traffic/flight_info/aeronav/Aero_Data/NASR_Subscription/) and put them in `./data/`
* Download the desired NASA Sherlock files from the [NASA website](https://sherlock.opendata.arc.nasa.gov/sherlock_open/) and place them in `./data/unprocessed/`

### Step 2. Parse and Convert Raw Data
```bash
python -m sherlock_reader ./data/unprocessed/ ./data/processed/ ./data/
```

### Step 3. Split Dataset
```bash
python data_splitter.py ../data/processed/ ../data/split
```

### Step 4. Train Model
```bash
python train.py --data ../data/split/ --output ./output-1/ --epochs 500
```

### Step 5a. Infer a Trajectory
```bash
python infer.py --model ./output-1/final-checkpoint.pth --data ../data/split/test.json --sample 0
```

### Step 5b. Evaluate the Model
```bash
python eval.py --model ./output-1/final-checkpoint.pth --data ../data/split/test.json
```

## Sample Model

A sample trained model checkpoint and small (GitHub-friendly) testing data JSON file are provided under `sample-trained-model/`. This model can be used with either the `infer.py` script or `eval.py` script to perform inference or model evaluation, respectively.

## Authors
Developed for CS 555 (Spring 2025) as the final project.