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
        Calculates the width in degrees of longitude
        """
        return self.east - self.west
    
    def height(self) -> float:
        """
        Calculates the height in degrees of latitude
        """
        return self.north - self.south
    
    def point_is_in(self, point: Point) -> bool:
        return point.longitude >= self.west and point.longitude <= self.east and point.latitude >= self.south and point.latitude <= self.north
    
    @staticmethod
    def from_dict(bbox: dict):
        return BoundingBox(bbox['north'], bbox['south'], bbox['east'], bbox['west'])
    
    def plot(self, plot):
        """
        Plot the grid using Matplotlib
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
        top_left = Point.deserialize(json_dict['top_left'])
        bottom_right = Point.deserialize(json_dict['bottom_right'])

        return GridSquare(top_left, bottom_right)
    
    def __repr__(self) -> str:
        return f'Square(top_left: {self.top_left}, bottom_right: {self.bottom_right})'
    
    def __str__(self) -> str:
        return repr(self)
    
    def to_rectangle(self) -> Rectangle:
        """
        Returns a Matplotlib Rectangle representing this Box which can be directly plotted
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
        with open(path, 'r') as f:
            return Grid.deserialize(json.load(f))

    @staticmethod
    def deserialize(json_dict: dict):
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
        Plot the grid using Matplotlib
        """
        for row in self.rows:
            for square in row:
                if square is not None:
                    plot.add_patch(square.to_rectangle())