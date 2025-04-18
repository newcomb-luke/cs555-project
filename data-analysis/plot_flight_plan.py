#=====================================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: Plots flight plans and actual trajectories from Sherlock IFF data over a map and grid
#=====================================================================================================

import argparse
import os
import sys
import matplotlib.pyplot as plt
from grid_input import Grid, Point, BoundingBox
from map import Map

# This will allow us to access sherlock-reader
sys.path.append('../')
from sherlock_reader import read_flights, FlightPlanParser


def main():
    parser = argparse.ArgumentParser(
        prog='Flight Plan Plotter',
        description='Reads in a flight plan converted to lat and long coordiantes and plots the route on a map'
    )
    parser.add_argument('input_path', help='The input flight data .csv format')
    parser.add_argument('faa_data_path', help='The directory path to the FAA data containing current airports, fixes, airways, etc.')
    parser.add_argument('-n', '--number', default="1", help='The index of the flight to start plotting at')
    parser.add_argument('-a', '--amount', default="10", help='The number of flights after the index to plot')

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f'Provided input path: `{args.input_path}` does not exist')
        exit(1)

    if not os.path.exists(args.faa_data_path):
        print(f'Provided FAA data path: `{args.faa_data_path}` does not exist')
        exit(2)
    
    number = None
    
    try:
        number = int(args.number)
    except ValueError:
        print(f'Provided flight index `{args.number}` is not a valid number')
        exit(2)

    amount = None
    
    try:
        amount = int(args.amount)
    except ValueError:
        print(f'Provided number of flights `{args.amount}` is not a valid number')
        exit(2)
    
    grid = Grid.deserialize_from_file('grid-definition.json')
    us_map = Map.deserialize_from_file('coordinates.json')
    
    fig, ax = plt.subplots()
    
    us_map.plot(ax)
    grid.plot(ax)

    parser = FlightPlanParser(args.faa_data_path)

    flight_data = []

    # Read the flights from the Sherlock file
    with read_flights(args.input_path) as flights:
        for i, flight in enumerate(flights):
            if i < number:
                continue

            if i > number + amount:
                break

            header = flight.header

            if '?' == header.dest or 'unassigned' == header.dest or \
                '?' == header.origin or 'unassigned' == header.origin:
                continue
            
            raw_flight_plans = flight.flight_plan

            first_plan = raw_flight_plans[0]

            parsed = parser.parse(first_plan.route)

            trajectory = []

            for point in flight.track_points:
                trajectory.append((point.coord_1, point.coord_2))
            
            flight_data.append({
                'trajectory': trajectory,
                'plan': parsed
            })

    for flight in flight_data:
        trajectory = flight['trajectory']
        plan = flight['plan']

        expanded = plan.expand()
        lat_long = expanded.to_lat_long()

        lat_long = prune_lat_long(lat_long, grid.bounding_box)
        trajectory = prune_lat_long(trajectory, grid.bounding_box)

        plot_flight_plan(ax, lat_long, lines=False)

        plot_flight_trajectory(ax, trajectory)

    plt.show()


def prune_lat_long(lat_long: list[tuple[float, float]], bounds: BoundingBox) -> list[tuple[float, float]]:
    """
    Removes coordinates outside the bounding box.

    Args:
        lat_long (list): List of (lat, lon) pairs.
        bounds (BoundingBox): Bounding area to keep points within.

    Returns:
        list: Filtered coordinates inside bounding box.
    """

    new_list = []

    for p in lat_long:
        point = Point(p[0], p[1])

        if bounds.point_is_in(point):
            new_list.append(p)

    return new_list


def plot_flight_trajectory(plot, trajectory, lines:bool=True):
    """
    Plots a flight trajectory (track points) on the map.

    Args:
        plot: Matplotlib axes object.
        trajectory: List of (lat, lon) coordinates.
        lines (bool): True for line plot, False for scatter.
    """

    points_x = [point[1] for point in trajectory] # Longitude
    points_y = [point[0] for point in trajectory] # Latitude

    if lines:
        plot.plot(points_x, points_y, alpha=0.5)
    else:
        plot.scatter(points_x, points_y, alpha=0.5)


def plot_flight_plan(plot, plan, lines:bool=True):
    """
    Plots an expanded flight plan (parsed route).

    Args:
        plot: Matplotlib axes object.
        plan: List of (lat, lon) coordinates from a parsed plan.
        lines (bool): True for line plot, False for scatter.
    """

    points_x = [point[1] for point in plan] # Longitude
    points_y = [point[0] for point in plan] # Latitude

    if lines:
        plot.plot(points_x, points_y, alpha=0.5)
    else:
        plot.scatter(points_x, points_y, alpha=0.5)


if __name__ == '__main__':
    main()