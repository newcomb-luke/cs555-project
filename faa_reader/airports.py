#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): Luke, Kayla
# Description: Reads and represents airport data from FAA-provided source files
#===============================================================================================

import pandas as pd


class Airport:
    """
    Represents a single airport with identifiers and location data.
    """

    def __init__(self, unique_id: str, airport_id: str, icao_id: str, name: str, latitude: float, longitude: float):
        self.unique_id = unique_id
        self.airport_id = airport_id
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.icao_id = icao_id
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Airport {{ no: {self.unique_id}, id: {self.airport_id}, icao: {self.icao_id}, name: {self.name}, lat: {self.latitude}, lon: {self.longitude} }}'


class AirportCollection:
    """
    Stores and indexes Airport objects for lookup by both FAA ID and ICAO ID.
    """

    def __init__(self):
        self.airport_id_map = {}
        self.icao_id_map = {}

    def add_airport(self, airport: Airport):
        """
        Adds a new airport to the collection.

        Args:
            airport (Airport): The airport to add.
        """

        self.airport_id_map[airport.airport_id] = airport

        if airport.icao_id is not None:
            self.icao_id_map[airport.icao_id] = airport
    
    def get_by_id(self, airport_id: str) -> Airport | None:
        """
        Looks up an airport by FAA identifier.

        Args:
            airport_id (str): The FAA ARPT_ID.

        Returns:
            Airport or None if not found.
        """

        return self.airport_id_map.get(airport_id)
    
    def get_by_icao(self, icao_id: str) -> Airport | None:
        """
        Looks up an airport by ICAO identifier.

        Args:
            icao_id (str): The ICAO_ID.

        Returns:
            Airport or None if not found.
        """

        return self.icao_id_map.get(icao_id)
    
    def get_by_either(self, id: str) -> Airport | None:
        """
        Tries to find an airport by either ICAO or FAA identifier.

        Args:
            id (str): The identifier to search.

        Returns:
            Airport or None if not found.
        """

        airport = self.get_by_icao(id)

        if airport is None:
            airport = self.get_by_id(id)
        
        return airport


class AirportsReader:
    """
    Reads an airport dataset from CSV and returns a populated AirportCollection.
    """

    def __init__(self):
        pass

    def read_airports(self, path: str) -> AirportCollection:
        """
        Parses a CSV file and constructs a collection of airports.

        Args:
            path (str): Path to the APT_BASE.csv file.

        Returns:
            AirportCollection: Indexed airport objects.
        """

        collection = AirportCollection()

        # We need to specify data types because pandas gets a little confused with those columns
        csv_rows = pd.read_csv(path, dtype={'ALT_FSS_ID': 'string', 'ALT_FSS_NAME': 'string', 'ALT_TOLL_FREE_NO': 'string', 'ICAO_ID': 'string'})

        for _, row in csv_rows.iterrows():
                unique_id = row['SITE_NO']
                airport_id = row['ARPT_ID']
                name = row['ARPT_NAME']
                latitude = float(row['LAT_DECIMAL'])
                longitude = float(row['LONG_DECIMAL'])
                icao_id = row['ICAO_ID']

                if pd.isna(icao_id):
                    icao_id = None

                airport = Airport(unique_id, airport_id, icao_id, name, latitude, longitude)

                collection.add_airport(airport)
        
        return collection


if __name__ == '__main__':
    airports_reader = AirportsReader()
    airports = airports_reader.read_airports('../data/APT_BASE.csv')

    print(airports.get_by_id('I66'))
    print(airports.get_by_icao('KJFK'))