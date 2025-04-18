#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: Visualizes flight trajectories from training data overlaid on a grid and map
#===============================================================================================

import argparse
import os
import json
import matplotlib.pyplot as plt
from grid_input import Grid
from map import Map


def main():
    parser = argparse.ArgumentParser(
        prog='Data Plotter',
        description='Reads data from the training data format and plots trajectories on a map'
    )
    parser.add_argument('input_path', help='The input training data in .json format')
    parser.add_argument('-n', '--number', default="1", help='The index of the flight to start plotting at')
    parser.add_argument('-a', '--amount', default="10", help='The number of flights after the index to plot')

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f'Provided input path: `{args.input_path}` does not exist')
        exit(1)
    
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
    
    trajectories = []

    with open(args.input_path, 'r') as f:
        data = json.load(f)

        sliced = data['e'][number:number+amount]

        for e in sliced:
            points = e['p']

            trajectory = [list(map(float, p.split(","))) for p in points]

            trajectories.append(trajectory)
    
    grid = Grid.deserialize_from_file('grid-definition.json')
    us_map = Map.deserialize_from_file('coordinates.json')
    
    fig, ax = plt.subplots()
    
    us_map.plot(ax)
    grid.plot(ax)

    for trajectory in trajectories:
        plot_trajectory(ax, trajectory)

    plt.show()


def plot_trajectory(plot, points, lines:bool=True):
    """
    Plots a single trajectory on the given Matplotlib plot.

    Args:
        plot: Matplotlib axes object.
        points (list): List of [lat, lon, alt, vx, vy, vz] for a single trajectory.
        lines (bool): Whether to draw lines (True) or points only (False).
    """

    points_x = [point[1] for point in points] # Longitude
    points_y = [point[0] for point in points] # Latitude

    if lines:
        plot.plot(points_x, points_y, alpha=0.5)
    else:
        plot.scatter(points_x, points_y, alpha=0.5)


if __name__ == '__main__':
    main()