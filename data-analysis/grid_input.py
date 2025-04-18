#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: Defines grid-based airspace partitioning for plotting and bounding checks
#===============================================================================================

import json
import math
from matplotlib.patches import Rectangle


class Point:
    """
    A single point defined by a latitude and longitude
    """

    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude
    
    @staticmethod
    def deserialize(json_dict: dict):
        """
        Deserializes a Point from a dictionary.

        Args:
            json_dict (dict): Dictionary with keys 'latitude' and 'longitude'.

        Returns:
            Point: Deserialized Point object.
        """

        latitude = json_dict['latitude']
        longitude = json_dict['longitude']

        return Point(latitude, longitude)
    
    def __repr__(self) -> str:
        return f'Lat {self.latitude}°, Lon {self.longitude}°'
    
    def __str__(self) -> str:
        return repr(self)


class BoundingBox:
    """
    This class represents a bounding box which is defined using latitudes and longitudes
    """
    def __init__(self, north: float, south: float, east: float, west: float):
        self.north = north
        self.south = south
        self.east = east
        self.west = west
    
    def to_rectangle(self) -> Rectangle:
        """
        Returns a Matplotlib Rectangle representing this BoundingBox which can be directly plotted
        """
        bottom_left = (self.west, self.south)
        width = self.width()
        height = self.height()
        # Matplotlib rectangles are defined by the bottom left point and are drawn upwards from there
        return Rectangle(bottom_left, width, height, fill=False)
    
    def width(self) -> float:
        """
        Calculates the width of the bounding box in degrees longitude.

        Returns:
            float: Width in degrees.
        """
        return self.east - self.west
    
    def height(self) -> float:
        """
        Calculates the height of the bounding box in degrees latitude.

        Returns:
            float: Height in degrees.
        """
        return self.north - self.south
    
    def point_is_in(self, point: Point) -> bool:
        """
        Checks whether a point lies inside the bounding box.

        Args:
            point (Point): The point to check.

        Returns:
            bool: True if the point is inside the box.
        """
        return point.longitude >= self.west and point.longitude <= self.east and point.latitude >= self.south and point.latitude <= self.north
    
    @staticmethod
    def from_dict(bbox: dict):
        """
        Deserializes a BoundingBox from a dictionary.

        Args:
            bbox (dict): Dictionary with 'north', 'south', 'east', 'west'.

        Returns:
            BoundingBox: The constructed bounding box.
        """

        return BoundingBox(bbox['north'], bbox['south'], bbox['east'], bbox['west'])
    
    def plot(self, plot):
        """
        Adds this bounding box to a Matplotlib plot.

        Args:
            plot: A Matplotlib axes instance.
        """
        plot.add_patch(self.to_rectangle())


class GridSquare:
    """
    A single square in our airspace grid
    """

    def __init__(self, top_left: Point, bottom_right: Point):
        self.top_left = top_left
        self.bottom_right = bottom_right
    
    @staticmethod
    def deserialize(json_dict: dict):
        """
        Deserializes a GridSquare from a dictionary.

        Args:
            json_dict (dict): Must contain 'top_left' and 'bottom_right' keys.

        Returns:
            GridSquare: The deserialized square.
        """

        top_left = Point.deserialize(json_dict['top_left'])
        bottom_right = Point.deserialize(json_dict['bottom_right'])

        return GridSquare(top_left, bottom_right)
    
    def __repr__(self) -> str:
        return f'Square(top_left: {self.top_left}, bottom_right: {self.bottom_right})'
    
    def __str__(self) -> str:
        return repr(self)
    
    def to_rectangle(self) -> Rectangle:
        """
        Converts the grid square to a Matplotlib Rectangle.

        Returns:
            Rectangle: Rectangle representation.
        """
        bottom_left = (self.top_left.longitude, self.bottom_right.latitude)
        width = math.fabs(self.top_left.longitude - self.bottom_right.longitude)
        height = math.fabs(self.top_left.latitude - self.bottom_right.latitude)
        # Matplotlib rectangles are defined by the bottom left point and are drawn upwards from there
        return Rectangle(bottom_left, width, height, fill=False, alpha=0.1)


class Grid:
    """
    The full airspace grid made up of rows of GridSquares
    """

    def __init__(self, rows: list[list[GridSquare]], bounding_box: BoundingBox):
        self.rows = rows
        self.bounding_box = bounding_box
    
    @staticmethod
    def deserialize_from_file(path: str):
        """
        Deserializes a Grid object from a JSON file.

        Args:
            path (str): Path to the JSON grid file.

        Returns:
            Grid: Deserialized Grid object.
        """

        with open(path, 'r') as f:
            return Grid.deserialize(json.load(f))

    @staticmethod
    def deserialize(json_dict: dict):
        """
        Deserializes a Grid from a JSON-compatible dictionary.

        Args:
            json_dict (dict): JSON structure with rows of squares and a bounding box.

        Returns:
            Grid: Reconstructed Grid object.
        """

        rows = []

        for json_row in json_dict['rows']:
            row = []
            for json_square in json_row:
                if json_square is not None:
                    square = GridSquare.deserialize(json_square)
                    row.append(square)
                else:
                    row.append(None)
            rows.append(row)
        
        bounding_box = BoundingBox.from_dict(json_dict['bounding_box'])
        
        return Grid(rows, bounding_box)
    
    def plot(self, plot):
        """
        Adds all non-empty grid squares to a Matplotlib plot.

        Args:
            plot: A Matplotlib axes instance.
        """
        for row in self.rows:
            for square in row:
                if square is not None:
                    plot.add_patch(square.to_rectangle())