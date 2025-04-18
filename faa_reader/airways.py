#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: Reads and represents airways from FAA data
#===============================================================================================

import pandas as pd

class Airway:
    """
    Represents a single airway with an identifier and its route sequence.
    """

    def __init__(self, airway_id: str, route: list[str]):
        self.airway_id = airway_id
        self.route = route
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Airway {{ id: {self.airway_id}, route: {self.route} }}'


class AirwayCollection:
    """
    Stores and indexes multiple Airway objects for lookup.
    """

    def __init__(self):
        self.airway_id_map = {}

    def add_airway(self, airway: Airway):
        """
        Adds a new Airway to the collection.

        Args:
            airway (Airway): The airway to add.
        """

        self.airway_id_map[airway.airway_id] = airway
    
    def get_by_id(self, airway_id: str) -> Airway | None:
        """
        Looks up an Airway by its identifier.

        Args:
            airway_id (str): The AWY_ID to search for.

        Returns:
            Airway or None if not found.
        """

        return self.airway_id_map.get(airway_id)


class AirwaysReader:
    """
    Reads an airway dataset from CSV and returns a populated AirwayCollection.
    """

    def __init__(self):
        pass
    
    def read_airways(self, path: str) -> AirwayCollection:
        """
        Parses a CSV file and constructs a collection of airways.

        Args:
            path (str): Path to the AWY_BASE.csv file.

        Returns:
            AirwayCollection: Indexed airway definitions.
        """

        collection = AirwayCollection()

        # We need to specify data types because pandas gets a little confused with those columns
        csv_rows = pd.read_csv(path)

        for _, row in csv_rows.iterrows():
                airway_id = row['AWY_ID']
                airway_string = row['AIRWAY_STRING']

                route = airway_string.split()

                airway = Airway(airway_id, route)

                collection.add_airway(airway)

        return collection


if __name__ == '__main__':
    airways_reader = AirwaysReader()
    airways = airways_reader.read_airways('../data/AWY_BASE.csv')

    print(airways.get_by_id('J10'))
    print(airways.get_by_id('J60'))
    print(airways.get_by_id('J80'))