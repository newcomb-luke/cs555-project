#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): Luke, Alex
# Description: Visualizes flight data as time-series signals accross the flight
#===============================================================================================

import argparse
import os
import sys
import matplotlib.pyplot as plt

# This will allow us to access sherlock-reader
sys.path.append('../')
from sherlock_reader import read_flights, RawTrajectoryPoint, RawTrajectory, TrajectoryPoint


def main():
    parser = argparse.ArgumentParser(
        prog='Raw Data Visualizer',
        description='Reads data from the NASA Sherlock data format and plots the values of each part of a trajectory'
    )
    parser.add_argument('input_path', help='The input Sherlock IFF data in .csv format')
    parser.add_argument('-n', '--number', default="1", help='The index of the flight to plot')

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f'Provided input path: `{args.input_path}` does not exist')
        exit(1)
    
    number = None
    
    try:
        number = float(args.number)
    except ValueError:
        print(f'Provided flight index `{args.number}` is not a valid number')
        exit(2)
    
    trajectories = []

    # Read the flights from the Sherlock file
    with read_flights(args.input_path) as flights:
        for i, flight in enumerate(flights):
            if i != number:
                continue

            flight_key = flight.header.flt_key

            raw_points = []
            for track_point in flight.track_points:
                raw_point = RawTrajectoryPoint(
                    track_point.rec_time,
                    track_point.coord_2,
                    track_point.coord_1,
                    track_point.alt,
                    track_point.ground_speed,
                    track_point.course,
                    track_point.rate_of_climb
                )

                # In some of the data it is missing certain fields, we have decided for the time being to just skip those
                if raw_point.has_all_fields():
                    raw_points.append(raw_point)
            
            raw_trajectory = RawTrajectory(raw_points)

            new_points = []

            for raw_point in raw_trajectory.points:
                point = TrajectoryPoint.from_raw_point(raw_point)

                new_point = {
                    'time': raw_point.timestamp,
                    'point': point
                }

                new_points.append(new_point)
            
            trajectories.append({
                'key': flight_key,
                'plan': flight.flight_plan,
                'points': new_points
            })

            break
    
    for trajectory in trajectories:
        points = trajectory['points']
        key = trajectory['key']

        start_time = points[0]['time']

        relative_times = []
        latitudes = []
        longitudes = []
        altitudes = []
        speeds_x = []
        speeds_y = []
        speeds_z = []

        for point in points:
            relative_time = point['time'] - start_time
            relative_times.append(relative_time)

            latitudes.append(point['point'].latitude)
            longitudes.append(point['point'].longitude)
            altitudes.append(point['point'].altitude)
            speeds_x.append(point['point'].speed_x)
            speeds_y.append(point['point'].speed_y)
            speeds_z.append(point['point'].speed_z)
        
        # plt.plot(relative_times, latitudes, label = f'{key} lat')
        # plt.plot(relative_times, longitudes, label = f'{key} lon')
        # plt.plot(relative_times, altitudes, label = f'{key} alt')
        plt.plot(relative_times, speeds_x, label = f'{key} spdx')
        # plt.plot(relative_times, speeds_y, label = f'{key} spdy')
        # plt.plot(relative_times, speeds_z, label = f'{key} spdz')

    # plt.legend()
    plt.show()


if __name__ == '__main__':
    main()