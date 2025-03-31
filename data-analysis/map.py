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
        with open(path, 'r') as f:
            return Map.deserialize(json.load(f))
    
    @staticmethod
    def deserialize(json_dict: dict):
        points = []

        raw_coordinates = json_dict['coordinates']
        for coordinate in raw_coordinates:
            # The format is for some reason [longitude, latitude]
            point = Point(coordinate[1], coordinate[0])
            points.append(point)
        
        return Map(points)
    
    def plot(self, plot, lines: bool=True):
        # Get the points as separate lists of x and y coordinates for matplotlib
        points_x = [point.longitude for point in self.points]
        points_y = [point.latitude for point in self.points]

        if lines:
            # Draw lines between the points
            plot.plot(points_x, points_y, alpha=0.5, c = '#000000')
        else:
            # Plot the individual points
            plot.scatter(points_x, points_y, alpha=0.5, c = '#000000')