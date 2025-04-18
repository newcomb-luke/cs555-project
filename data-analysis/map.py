#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): Luke, Alex
# Description: Loads and plots map outlines from coordinate-based JSON files
#===============================================================================================

import json
from grid_input import Point


class Map:
    """
    A map defined as a set of points
    """

    def __init__(self, points: list[Point]):
        self.points = points
    
    @staticmethod
    def deserialize_from_file(path: str):
        """
        Loads a Map object from a JSON file.

        Args:
            path (str): File path to a JSON file containing coordinate data.

        Returns:
            Map: Deserialized Map object.
        """

        with open(path, 'r') as f:
            return Map.deserialize(json.load(f))
    
    @staticmethod
    def deserialize(json_dict: dict):
        """
        Parses a dictionary to construct a Map.

        Args:
            json_dict (dict): Dictionary with 'coordinates' as list of [lon, lat] pairs.

        Returns:
            Map: The reconstructed map.
        """

        points = []

        raw_coordinates = json_dict['coordinates']
        for coordinate in raw_coordinates:
            # The format is for some reason [longitude, latitude]
            point = Point(coordinate[1], coordinate[0])
            points.append(point)
        
        return Map(points)
    
    def plot(self, plot, lines: bool=True):
        """
        Plots the map onto a given Matplotlib axes.

        Args:
            plot: Matplotlib axes object to draw onto.
            lines (bool): Whether to draw lines between points (True) or scatter points only (False).
        """

        # Get the points as separate lists of x and y coordinates for matplotlib
        points_x = [point.longitude for point in self.points]
        points_y = [point.latitude for point in self.points]

        if lines:
            # Draw lines between the points
            plot.plot(points_x, points_y, alpha=0.5, c = '#000000')
        else:
            # Plot the individual points
            plot.scatter(points_x, points_y, alpha=0.5, c = '#000000')