#=====================================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: Parses and validates Sherlock flight data for model training with FAA-provided context
#=====================================================================================================

import argparse
import os
import sys
import json
from .data_reader import read_flights, Flight
from .convert import RawTrajectoryPoint, RawTrajectory, convert_trajectory, TrajectoryPoint
from .flight_plan import FlightPlanParser

# This will allow us to access faa_reader
sys.path.append('../')
from faa_reader import Airport, AirportCollection, AirportsReader


class ValidationConfig:
    """
    Stores filtering configuration for trajectory preprocessing.
    """
    def __init__(self, interval: float, min_altitude: float, min_num_points: int):
        self.interval = interval
        self.min_altitude = min_altitude
        self.min_num_points = min_num_points


def main():
    parser = argparse.ArgumentParser(
        prog='Sherlock Data Reader',
        description='Converts data from the NASA Sherlock IFF format and outputs a continuous full output file to be used for model training'
    )
    parser.add_argument('input_path', help='The input Sherlock IFF data in .csv format')
    parser.add_argument('output_path', help='The file path to output continuous full training data into one .json file per input file')
    parser.add_argument('faa_data_path', help='The directory path to the FAA data containing current airports, fixes, airways, etc.')
    parser.add_argument('-i', '--interval', default="120", help='The interval at which to output data, in seconds. For every 2 minutes, the value is 120')
    parser.add_argument('-a', '--min-altitude', default="18000", help='The minimum altitude for points to be included in the trajectory')
    parser.add_argument('-n', '--min-num-points', default="10", help='The minimum number of points in a valid trajectory')

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f'Provided input path: `{args.input_path}` does not exist')
        exit(1)

    if not os.path.exists(args.faa_data_path):
        print(f'Provided FAA data path: `{args.faa_data_path}` does not exist')
        exit(2)
    
    interval = check_float_arg(args.interval, 'interval')
    min_altitude = check_float_arg(args.min_altitude, 'minimum altitude')
    min_num_points = check_int_arg(args.min_num_points, 'minimum number of points')

    config = ValidationConfig(interval, min_altitude, min_num_points)

    airports_path = os.path.join(args.faa_data_path, 'APT_BASE.csv')
    airports_reader = AirportsReader()
    airports = airports_reader.read_airports(airports_path)

    flight_plan_parser = FlightPlanParser(args.faa_data_path)

    input_file_paths = []
    output_file_paths = []

    if os.path.isdir(args.input_path):
        if not os.path.isdir(args.output_path):
            print(f'Input path `{args.input_path}` was a directory, however the output path `{args.output_path}` was not. If input was a directory, the output must be as well.')
            exit(6)

        for entry in os.scandir(args.input_path):
            if entry.is_file() and entry.name.endswith('csv'):
                input_file_paths.append(entry.path)

                base, _ = os.path.splitext(entry.name)
                output_path = os.path.join(args.output_path, base + '.json')
                output_file_paths.append(output_path)
    else:
        input_file_paths.append(args.input_path)

        if os.path.isdir(args.output_path):
            base, _ = os.path.splitext(os.path.basename(args.input_path))
            output_path = os.path.join(args.output_path, base + '.json')
            output_file_paths.append(output_path)
        else:
            output_file_paths.append(output_path)
    
    total_num_valid = 0
    total_num = 0
    
    for input_path, output_path in zip(input_file_paths, output_file_paths):
        if os.path.exists(output_path):
            print(f'File {input_path} was already processed, skipping')
            continue

        num_valid, total = filter_and_convert_data(input_path, output_path, airports, flight_plan_parser, config)

        total_num_valid += num_valid
        total_num += total
    
    print(f'Overall for batch: valid {total_num_valid} / {total_num} total')


def filter_and_convert_data(input_file_path: str, output_file_path: str, airports: AirportCollection, flight_plan_parser: FlightPlanParser, config: ValidationConfig) -> tuple[int, int]:
    """
    Filters and converts all flights in a file, saving valid ones as compact JSON.

    Args:
        input_file_path (str): Path to Sherlock CSV.
        output_file_path (str): Path for saving processed JSON.
        airports (AirportCollection): Reference airport database.
        flight_plan_parser (FlightPlanParser): Parses textual flight plans.
        config (ValidationConfig): Filtering parameters.

    Returns:
        tuple[int, int]: Count of valid flights and total flights processed.
    """

    with read_flights(input_file_path) as flights:
        entries = []
        num_valid_flights = 0
        num_total_flights = 0

        for flight in flights:
            num_total_flights += 1

            data = validate_flight(flight, airports, flight_plan_parser, config)

            if data is None:
                print(f'Found flight {flight.header.flt_key}...invalid')
                continue
            
            source_airport, dest_airport, waypoints, converted_points = data

            waypoint_strings = []
            for waypoint in waypoints:
                waypoint_string = f'{waypoint[0]},{waypoint[1]}'
                waypoint_strings.append(waypoint_string)
            
            point_strings = []
            for point in converted_points:
                point_string = f'{point.latitude},{point.longitude},{point.altitude},{point.speed_x},{point.speed_y},{point.speed_z}'
                point_strings.append(point_string)
            
            # This weird-ish format is meant to save space in the file, since it will be repeated a bunch of times for every entry
            entry = {
                's': f'{source_airport.latitude},{source_airport.longitude}',
                'w': waypoint_strings,
                'd': f'{dest_airport.latitude},{dest_airport.longitude}',
                'p': point_strings
            }

            print(f'Found flight {flight.header.flt_key}...valid')

            entries.append(entry)
            num_valid_flights += 1

        with open(output_file_path, 'w') as f:
            json.dump({'e': entries}, f)
        
        print(f'Wrote {num_valid_flights} valid flights, out of {num_total_flights} total flights')

        return num_valid_flights, num_total_flights


def validate_flight(flight: Flight, airports: AirportCollection, flight_plan_parser: FlightPlanParser, config: ValidationConfig) -> tuple[Airport, Airport, list[TrajectoryPoint]] | None:
    """
    Fully validates a single flight for training.

    Returns:
        tuple containing source, destination, waypoints, and converted trajectory points if valid, else None.
    """

    valid_airports = validate_flight_route(flight, airports)

    if valid_airports is None:
        return None
    
    source_airport, dest_airport = valid_airports

    # This is really sad, because it essentially filters by what airways and stuff changed from then
    # until we have our FAA data for, but this is the only way without doing a whole bunch of other stuff

    try:
        # Read flight plan
        raw_flight_plans = flight.flight_plan
        first_plan = raw_flight_plans[0]

        parsed = flight_plan_parser.parse(first_plan.route)
        expanded = parsed.expand()
        waypoints = expanded.to_lat_long()
    except:
        # Any exception means the flight plan couldn't be parsed, so we just throw it away :(
        return None

    points = validate_flight_points(flight, config)

    if points is None:
        return None
    
    return source_airport, dest_airport, waypoints, points


def validate_flight_points(flight: Flight, config: ValidationConfig) -> list[TrajectoryPoint] | None:
    """
    Filters and converts trajectory points for a flight.

    Returns:
        list of valid TrajectoryPoints or None if the count is below the configured threshold.
    """

    raw_points = []

    for track_point in flight.track_points:
        raw_point = RawTrajectoryPoint(
            track_point.rec_time,
            track_point.coord_1,
            track_point.coord_2,
            track_point.alt * 100.0,
            track_point.ground_speed,
            track_point.course,
            track_point.rate_of_climb
        )

        # In some of the data it is missing certain fields, we have decided for the time being to just skip those
        if raw_point.has_all_fields():
            raw_points.append(raw_point)

    converted_points = []

    for point in convert_trajectory(RawTrajectory(raw_points), config.interval):
        if point.altitude >= config.min_altitude:
            converted_points.append(point)
    
    if len(converted_points) < config.min_num_points:
        return None
    
    return converted_points


def validate_flight_route(flight: Flight, airports: AirportCollection) -> tuple[Airport, Airport] | None:
    """
    Validates the source and destination airports of a flight.

    Returns:
        Tuple of source and destination Airport objects, or None if invalid.
    """

    # Sherlock actually did some work to determine more or less the "actual" origin and destination airports of a flight
    # and they are considered "estimated". We get them from the header
    # We have debated if we should use those or not, and decided for the meantime to go with the "real" data we know, even if that means
    # that we need more of it
    source_airport_id = flight.header.origin
    dest_airport_id = flight.header.dest

    # I think mostly smaller planes have this set, ones that we don't really care about
    if source_airport_id == '?' or dest_airport_id == '?' or source_airport_id == 'unassigned' or dest_airport_id == 'unassigned':
        return None
    
    # Interestingly too, the airports we really care about are the international ones... which start with a 'K' if they are in ICAO format
    # If we want to include the smaller airports, we can remove this later
    if not source_airport_id.startswith('K') or not dest_airport_id.startswith('K'):
        return None
    
    # We only want KPHL to KMCO

    # if source_airport_id != 'KPHL' or dest_airport_id != 'KMCO':
    #     return None

    source_airport = check_faa_icao_airport(airports, source_airport_id)
    dest_airport = check_faa_icao_airport(airports, dest_airport_id)

    # This isn't useful for training
    if source_airport is None or dest_airport is None:
        return None
    
    return source_airport, dest_airport


def check_faa_icao_airport(airports: AirportCollection, airport_id: str) -> Airport | None:
    """
    Checks an AirportCollection for an airport first by ICAO id and then by FAA id, returning None if neither have the airport id

    Args:
        airports (AirportCollection): Airport lookup table.
        airport_id (str): ICAO or FAA ID to search.

    Returns:
        Airport or None if not found.
    """
    airport = airports.get_by_icao(airport_id)

    if airport is None:
        airport = airports.get_by_id(airport_id)
    
    return airport


def check_float_arg(value: str, name: str) -> float:
    """
    Validates and converts a command-line string to float.

    Args:
        value (str): The value to convert.
        name (str): Argument name (used in error messages).

    Returns:
        float: Converted float value.
    """
    try:
        return float(value)
    except ValueError:
        print(f'Provided value for {name} `{value}` is not a valid number')
        exit(3)


def check_int_arg(value: str, name: str) -> int:
    """
    Validates and converts a command-line string to int.

    Args:
        value (str): The value to convert.
        name (str): Argument name (used in error messages).

    Returns:
        int: Converted int value.
    """
    try:
        return int(value)
    except ValueError:
        print(f'Provided value for {name} `{value}` is not a valid number')
        exit(4)


if __name__ == "__main__":
    main()