#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: Classes and utilities for representing and converting raw flight trajectories
#===============================================================================================


import math
import numpy as np


def lerp(a: str, b: str, t: float) -> float:
    """
    Linearly interpolates between two numeric strings a and b by a ratio t.

    Args:
        a (str): The starting value.
        b (str): The ending value.
        t (float): Interpolation ratio in [0, 1].

    Returns:
        float: Interpolated value.
    """
    return ((float(b) - float(a)) * t) + float(a)


class RawTrajectoryPoint:
    """
    Represents a raw trajectory point extracted from CSV data.
    """
    def __init__(self, timestamp: float, latitude: float, longitude: float, altitude: float, ground_speed: float, course: float, rate_of_climb: float):
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.ground_speed = ground_speed
        self.course = course
        self.rate_of_climb = rate_of_climb
            
    @staticmethod
    def from_csv(line: list[str]):
        """
        Parses a raw trajectory point from a list of CSV string values.

        Args:
            line (list[str]): CSV fields representing a trajectory point.

        Returns:
            RawTrajectoryPoint: Parsed object.
        """
        separated = iter(line)

        timestamp = float(next(separated))
        latitude = float(next(separated))
        longitude = float(next(separated))
        altitude = float(next(separated))
        ground_speed = float(next(separated))
        course = float(next(separated))
        rate_of_climb = float(next(separated))

        return RawTrajectoryPoint(timestamp, latitude, longitude, altitude, ground_speed, course, rate_of_climb)
    
    def has_all_fields(self) -> bool:
        """
        Returns True if this trajectory point actually has all of its fields, or False if some are missing
        """
        return self.ground_speed and (self.course != -99) and (self.rate_of_climb != -99999)


class RawTrajectory:
    """
    Represents a sequence of raw trajectory points.
    """
    def __init__(self, points: list[RawTrajectoryPoint]):
        self.points = points


class TrajectoryPoint:
    """
    Represents a processed trajectory point including 3D position and velocity.
    """
    def __init__(self, latitude: float, longitude: float, altitude: float, speed_x: float, speed_y: float, speed_z: float):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.speed_z = speed_z
    
    def into_np_array(self):
        """
        Converts the trajectory point into a NumPy array.
        Longitude is first, then latitude, followed by altitude and velocities.

        Returns:
            np.ndarray: [lon, lat, alt, speed_x, speed_y, speed_z]
        """
        return np.array([self.longitude, self.latitude, self.altitude, self.speed_x, self.speed_y, self.speed_z])
    
    @staticmethod
    def _course_to_speeds(course: float, ground_speed: float) -> tuple[float, float]:
        """
        Converts course angle and ground speed to x/y speed components.

        Args:
            course (float): Direction of travel in degrees.
            ground_speed (float): Speed over the ground.

        Returns:
            tuple[float, float]: x and y components of speed.
        """
        x_speed = math.sin(math.radians(course)) * ground_speed
        y_speed = math.cos(math.radians(course)) * ground_speed

        return (x_speed, y_speed)
    
    @staticmethod
    def from_raw_point(point: RawTrajectoryPoint):
        """
        Converts a RawTrajectoryPoint into a processed TrajectoryPoint.

        Args:
            point (RawTrajectoryPoint): The raw point to convert.

        Returns:
            TrajectoryPoint: The processed version including velocity vectors.
        """
        latitude = point.latitude
        longitude = point.longitude
        altitude = point.altitude
        speed_x, speed_y = TrajectoryPoint._course_to_speeds(point.course, point.ground_speed)
        speed_z = point.rate_of_climb

        return TrajectoryPoint(latitude, longitude, altitude, speed_x, speed_y, speed_z)


class TrajectoryConverter:
    """
    Converts a raw trajectory into evenly spaced interpolated trajectory points.
    """

    def __init__(self, raw_trajectory: RawTrajectory, time_interval_seconds: int):
        self.raw_trajectory = iter(raw_trajectory.points)
        self.time_interval_seconds = time_interval_seconds
        self.previous_point = None
        self.next_point = None
        self.target_seconds = 0.0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> TrajectoryPoint:
        if self.previous_point is None:
            self.previous_point = next(self.raw_trajectory)

            self.target_seconds = self.previous_point.timestamp + self.time_interval_seconds
        
            self.next_point = next(self.raw_trajectory)

            # Return the starting point because it is valid
            return TrajectoryPoint.from_raw_point(self.previous_point)
        
        while self.target_seconds > self.next_point.timestamp:
            # We need to get the next point and use that
            self.previous_point = self.next_point
            self.next_point = next(self.raw_trajectory)
            # This doesn't change our target seconds or anything

        # We know that the next point is in the future, and the previous point is in the past
        time_into_interval = self.target_seconds - self.previous_point.timestamp
        ratio = time_into_interval / (self.next_point.timestamp - self.previous_point.timestamp)

        interpolated_point = self._lerp_points(self.previous_point, self.next_point, ratio)

        self.target_seconds += self.time_interval_seconds

        return interpolated_point

    def _lerp_points(self, point_1: RawTrajectoryPoint, point_2: RawTrajectoryPoint, ratio: float) -> TrajectoryPoint:
        """
        Interpolates between two raw trajectory points.

        Args:
            point_1 (RawTrajectoryPoint): The earlier point.
            point_2 (RawTrajectoryPoint): The later point.
            ratio (float): Interpolation ratio between the two.

        Returns:
            TrajectoryPoint: A new interpolated point.
        """
        converted_point_1 = TrajectoryPoint.from_raw_point(point_1)
        converted_point_2 = TrajectoryPoint.from_raw_point(point_2)

        latitude = lerp(converted_point_1.latitude, converted_point_2.latitude, ratio)
        longitude = lerp(converted_point_1.longitude, converted_point_2.longitude, ratio)
        altitude = lerp(converted_point_1.altitude, converted_point_2.altitude, ratio)
        speed_x = lerp(converted_point_1.speed_x, converted_point_2.speed_x, ratio)
        speed_y = lerp(converted_point_1.speed_y, converted_point_2.speed_y, ratio)
        speed_z = lerp(converted_point_1.speed_z, converted_point_2.speed_z, ratio)

        return TrajectoryPoint(latitude, longitude, altitude, speed_x, speed_y, speed_z)

def convert_trajectory(raw_trajectory: RawTrajectory, time_interval_seconds: float) -> TrajectoryConverter:
    """
    Converts a raw trajectory into a generator of interpolated points spaced by a fixed interval.

    Args:
        raw_trajectory (RawTrajectory): The original flight path data.
        time_interval_seconds (float): Time interval between points.

    Returns:
        TrajectoryConverter: An iterable generator for interpolated points.
    """
    return TrajectoryConverter(raw_trajectory, time_interval_seconds)