#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: Classes responsible for representing and reading aircraft flight plans
#===============================================================================================

import re
import sys
import copy
import os
from enum import Enum

# This will allow us to access faa_reader
sys.path.append('../')
from faa_reader import AirportsReader
from faa_reader import Airport as RealAirport
from faa_reader import FixesReader, Fix
from faa_reader import NavaidsReader
from faa_reader import Navaid as RealNavaid
from faa_reader import AirwaysReader
from faa_reader import Airway as RealAirway


class PlanType(Enum):
    """
    Enumeration of all possible flight plan item types.
    """
    AIRPORT = 0
    NAV_AID = 1
    AIRWAY = 2
    WAYPOINT = 3
    DIRECT_FIX = 4
    STAR = 5
    AIRPORT_ETA = 6
    LAT_LONG = 7
    UNKNOWN = 8


class PlanItem:
    """
    Abstract base class for all flight plan items.
    """

    def __init__(self):
        pass

    def type(self) -> PlanType:
        """
        Returns the type of the plan item.
        """
        raise NotImplemented()
    
    def __str__(self):
        return f"{self.type().name}: {self.__dict__}"

    def __repr__(self):
        return self.__str__()


class Airport(PlanItem):
    """
    Represents an airport in a flight plan.
    """
    def __init__(self, real: RealAirport):
        super(Airport, self).__init__()
        self.real = real
    
    def type(self) -> PlanType:
        return PlanType.AIRPORT
    
    def __str__(self):
        return f"Airport: {self.real.name}, id: {self.real.icao_id}"
    
    def __eq__(self, value):
        if not isinstance(value, Airport):
            return False
        
        return self.real.icao_id == value.real.icao_id


class Navaid(PlanItem):
    """
    Represents a navigational aid in a flight plan.
    """
    def __init__(self, real: RealNavaid):
        super(Navaid, self).__init__()
        self.real = real
    
    def type(self) -> PlanType:
        return PlanType.NAV_AID
    
    def __str__(self):
        return f"Navaid: {self.real.navaid_id}"
    
    def __eq__(self, value):
        if not isinstance(value, Navaid):
            return False
        
        return self.real.navaid_id == value.real.navaid_id


class Airway(PlanItem):
    """
    Represents an airway in a flight plan.
    """
    def __init__(self, real: RealAirway, route: list[PlanItem]):
        super(Airway, self).__init__()
        self.real = real
        self.route = route
    
    def type(self) -> PlanType:
        return PlanType.AIRWAY
    
    def __str__(self):
        return f"Airway: {self.real.airway_id}"
    
    def __eq__(self, value):
        if not isinstance(value, Airway):
            return False
        
        return self.real.airway_id == value.real.airway_id


class Waypoint(PlanItem):
    """
    Represents a generic waypoint in a flight plan.
    """
    def __init__(self, real: Fix):
        super(Waypoint, self).__init__()
        self.real = real
    
    def type(self) -> PlanType:
        return PlanType.WAYPOINT
    
    def __str__(self):
        return f"Waypoint: {self.real.fix_id}"
    
    def __eq__(self, value):
        if not isinstance(value, Waypoint):
            return False
        
        return self.real.fix_id == value.real.fix_id


class DirectFix(PlanItem):
    """
    An offset from a fix. A fix is a 3-letter location. The offset is radially, so the first 3-digit
    number after the 3 letters is the bearing, and the second 3-digit number represents the distance from it
    at the bearing given in nautical miles
    """
    def __init__(self, name: str, bearing: int, distance: int):
        super(DirectFix, self).__init__()
        self.name = name
        self.bearing = bearing
        self.distance = distance
    
    def type(self) -> PlanType:
        return PlanType.DIRECT_FIX
    
    def __str__(self):
        return f"Direct Fix: {self.name}, bearing: {self.bearing}°, distance: {self.distance} NM"


class Star(PlanItem):
    """
    STAR - Standard Terminal Arrival Route
    This represents the route that an aircraft will follow to land at a destination airport
    """
    def __init__(self, name: str, revision: str):
        super(Star, self).__init__()
        self.name = name
        self.revision = revision
    
    def type(self) -> PlanType:
        return PlanType.STAR
    
    def __str__(self):
        return f"STAR: {self.name}-{self.revision}"


class AirportEta(PlanItem):
    """
    ETA - Estimated Time of Arrival
    This represents the airport that it will land at, as well as the landing time in UTC
    """
    def __init__(self, real: RealAirport, time_hours: int, time_minutes: int):
        super(AirportEta, self).__init__()
        self.real = real
        self.time_hours = time_hours
        self.time_minutes = time_minutes
    
    def type(self) -> PlanType:
        return PlanType.AIRPORT_ETA
    
    def __str__(self):
        return f"ETA: {self.real.airport_id} at {self.time_hours:02d}:{self.time_minutes:02d} UTC"


class LatLong(PlanItem):
    """
    This represents a fix using lat/long coordinates
    """
    def __init__(self, lat: str, lon: str):
        super(LatLong, self).__init__()
        self.lat = lat
        self.lon = lon
    
    def type(self) -> PlanType:
        return PlanType.LAT_LONG
    
    def __str__(self):
        return f"LatLon: {self.lat}/{self.lon}"


class Unknown(PlanItem):
    """
    Hopefully we have none of these
    """
    def __init__(self, text: str):
        super(Unknown, self).__init__()
        self.text = text
    
    def type(self) -> PlanType:
        return PlanType.UNKNOWN
    
    def __str__(self):
        return f"Unknown Element: {self.text}"


class FlightPlan:
    """
    Represents a parsed and possibly expanded flight plan.
    """
    def __init__(self, items: list[PlanItem]):
        self._items = items
    
    def items(self) -> list[PlanItem]:
        """
        Returns the list of items in the flight plan.
        """
        return self._items
    
    def expand(self):
        """
        Expands airways in the flight plan into their constituent fixes.
        """
        expanded = []

        for i, item in enumerate(self._items):
            if item.type() == PlanType.AIRWAY:
                before_airway = self._items[i - 1]
                after_airway = self._items[i + 1]

                airway_segment = self._airway_segments_between(before_airway, after_airway, item)

                expanded.extend(airway_segment)
            else:
                expanded.append(copy.deepcopy(item))

        return FlightPlan(expanded)
    
    def _airway_segments_between(self, before: PlanItem, after: PlanItem, airway: Airway) -> list[PlanItem]:
        """
        Returns the segment of the airway between two points.
        """
        forward, index_before, index_after = self._airway_segment_positions(before, after, airway)

        if forward:
            segments = airway.route[index_before+1:index_after-1]
        else:
            segments = airway.route[index_before-1:index_after+1:-1]
        
        return segments
    
    def _airway_segment_positions(self, before: PlanItem, after: PlanItem, airway: Airway) -> tuple[bool, int, int]:
        """
        Finds the positions of two points in an airway and determines the direction.
        """
        try:
            index_before = airway.route.index(before)
        except:
            raise Exception(f'Attempted to find location of fix/waypoint in airway route that doesn\'t exist: Start: {before}, End: {after}, Airway: {airway.route}')

        try:
            index_after = airway.route.index(after)
        except:
            raise Exception(f'Attempted to find location of fix/waypoint in airway route that doesn\'t exist: Start: {before}, End: {after}, Airway: {airway.route}')
        
        return index_before < index_after, index_before, index_after
    
    def to_lat_long(self) -> list[tuple[float, float]]:
        """
        Converts all plan items to a list of latitude and longitude coordinates.
        """
        points = []

        for item in self.items():
            item_points = self._item_to_lat_long(item)
            points.extend(item_points)

        return points
    
    def _item_to_lat_long(self, item: PlanItem) -> list[tuple[float, float]]:
        """
        Helper to extract lat/long from a single item.
        """
        if item.type() == PlanType.AIRPORT:
            lat = item.real.latitude
            long = item.real.longitude
            return [(lat, long)]
        elif item.type() == PlanType.AIRPORT_ETA:
            lat = item.real.latitude
            long = item.real.longitude
            return [(lat, long)]
        elif item.type() == PlanType.AIRWAY:
            raise Exception('Airway elements should have been expanded in earlier function call')
        elif item.type() == PlanType.LAT_LONG:
            lat = item.lat
            long = item.lon
            return [(lat, long)]
        elif item.type() == PlanType.NAV_AID:
            lat = item.real.latitude
            long = item.real.longitude
            return [(lat, long)]
        elif item.type() == PlanType.WAYPOINT:
            lat = item.real.latitude
            long = item.real.longitude
            return [(lat, long)]
        elif item.type() == PlanType.DIRECT_FIX:
            raise Exception(f'Not yet implemented: Direct Fix {item}')
        elif item.type() == PlanType.STAR:
            # Skip this one, not an error
            return []
        elif item.type() == PlanType.UNKNOWN:
            # Skip anything we don't know really
            return []
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return f"Flight plan: {self._items}"


class FlightPlanParser:
    """
    Parses flight plan strings into structured flight plan objects.
    """
    def __init__(self, data_path: str):
        self.direct_fix_pattern = re.compile(r'\b[A-Z]{3}\d{3}\d{3}')  # Direct fixes (e.g., AIR097030)
        self.star_pattern = re.compile(r'([A-Z]{3,5})(\d)([A-Z]?)')  # STAR (e.g., RAZRR4, CHERI3)
        self.eta_pattern = re.compile(r'([A-Z]{4})/(\d{4})')  # ETA format (e.g., KSJC/0509)
        self.lat_lon_pattern = re.compile(r'\b\d{2,6}[NS]/\d{2,6}[EW]') # Lat/Long fix (e.g., 405812N/0740034W)
        self.waypoint_pattern = re.compile(r'\b[A-Z]{5}\b')  # Waypoints, or generic fixes (e.g., TWAIN, KNGRY)

        airports_path = os.path.join(data_path, 'APT_BASE.csv')
        fixes_path = os.path.join(data_path, 'FIX_BASE.csv')
        airways_path = os.path.join(data_path, 'AWY_BASE.csv')
        navaids_path = os.path.join(data_path, 'NAV_BASE.csv')

        airports_reader = AirportsReader()
        self.airports = airports_reader.read_airports(airports_path)

        fixes_reader = FixesReader()
        self.fixes = fixes_reader.read_fixes(fixes_path)

        airways_reader = AirwaysReader()
        self.airways = airways_reader.read_airways(airways_path)

        navaids_reader = NavaidsReader()
        self.navaids = navaids_reader.read_navaids(navaids_path)
    
    def parse(self, flight_plan: str) -> FlightPlan:
        """
        Parses a flight plan string into a structured FlightPlan.
        """
        split_elements = flight_plan.split('.')
        elements = [elem for elem in split_elements if elem and elem != '/']

        parsed = []
        
        parsed.append(self._parse_source(elements[0]))

        for elem in elements[1:-1]:
            # This has some meaning, but it is too complicated for what we want to do
            elem = elem.strip().replace('*', '')
            # This has a meaning too, but oh well
            elem = elem.strip().replace('+', '')

            parsed.append(self._element_to_item(elem))

        parsed.append(self._parse_destination(elements[-1]))
        
        return FlightPlan(parsed)
    
    def _parse_source(self, element: str) -> PlanItem:
        """
        Parses the first element as the source airport.
        """
        source = self.airports.get_by_either(element)

        if source is None:
            source = self._element_to_item(element)
        else:
            source = Airport(source)
        
        return source
    

    def _parse_destination(self, element: str) -> PlanItem:
        """
        Parses the last element as the destination airport or ETA.
        """
        dest = self.airports.get_by_either(element)

        if dest is None:
            if self.eta_pattern.match(element):
                airport_name = element[:4]
                airport = self.airports.get_by_either(airport_name)

                if airport is not None:
                    hours = int(element[5:7])
                    minutes = int(element[7:])
                    dest = AirportEta(airport, hours, minutes)
                else:
                    dest = self._element_to_item(element)
            else:
                dest = self._element_to_item(element)
        else:
            dest = Airport(dest)
            
        return dest

    
    def _element_to_item(self, element: str) -> PlanItem:
        """
        Converts an element string to a PlanItem.
        """
        if self.waypoint_pattern.match(element):
            fix = self.fixes.get_by_id(element)

            if fix is not None:
                return Waypoint(fix)
        
        airway = self.airways.get_by_id(element)

        if airway is not None:
            route = self._parse_airway_route(airway)
            return Airway(airway, route)
        
        navaid = self.navaids.get_by_id(element)

        if navaid is not None:
            return Navaid(navaid)
        
        if self.lat_lon_pattern.match(element):
            split = element.split('/')
            lat = split[0]
            lon = split[1]
            return LatLong(lat, lon)

        if self.direct_fix_pattern.match(element):
            fix_name = element[:3]

            fix = self.fixes.get_by_id(element)

            if fix is not None:
                fix_bearing = int(element[3:6])
                fix_distance = int(element[6:])
                return DirectFix(fix_name, fix_bearing, fix_distance)

        match = self.star_pattern.match(element)
        if match:
            name = match.group(1)
            number = match.group(2)
            suffix = match.group(3)
            return Star(name, number + suffix)

        return Unknown(element)
    
    def _parse_airway_route(self, airway: RealAirway) -> list[PlanItem]:
        """
        Parses the internal route of an airway.
        """
        route = []

        for element in airway.route:
            if self.waypoint_pattern.match(element):
                fix = self.fixes.get_by_id(element)

                if fix is not None:
                    route.append(Waypoint(fix))
                    continue
            
            navaid = self.navaids.get_by_id(element)

            if navaid is not None:
                route.append(Navaid(navaid))
                continue

            raise Exception(f'Unrecognized element in airway route: {element}')

        return route
    

if __name__ == '__main__':
    flight_plan = 'KLGA..NEWEL.J60.IOW.J10.DSM..OBH..BRWRY.LAWGR3.KDEN/0500'
    parser = FlightPlanParser()
    parsed_plan = parser.parse(flight_plan)

    print(parsed_plan)

    expanded = parsed_plan.expand()

    print(expanded)

    lat_long = expanded.to_lat_long()

    print(lat_long)