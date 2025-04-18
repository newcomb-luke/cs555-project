#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: Reads and represents fixes (waypoints) from FAA data
#===============================================================================================

import pandas as pd

class Fix:
    """
    Represents a single enroute fix (waypoint) with an identifier and geographic coordinates.
    """

    def __init__(self, fix_id: str, latitude: float, longitude: float):
        self.fix_id = fix_id
        self.latitude = latitude
        self.longitude = longitude
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Fix {{ id: {self.fix_id}, lat: {self.latitude}, lon: {self.longitude} }}'


class FixCollection:
    """
    Stores and indexes multiple Fix objects for lookup.
    """

    def __init__(self):
        self.fix_id_map = {}

    def add_fix(self, fix: Fix):
        """
        Adds a new Fix to the collection.

        Args:
            fix (Fix): The fix to add.
        """
        self.fix_id_map[fix.fix_id] = fix
    
    def get_by_id(self, fix_id: str) -> Fix | None:
        """
        Looks up a Fix by its identifier.

        Args:
            fix_id (str): The FIX_ID to search for.

        Returns:
            Fix or None if not found.
        """
        return self.fix_id_map.get(fix_id)


class FixesReader:
    """
    Reads a FIX dataset from CSV and returns a populated FixCollection.
    """

    def __init__(self):
        pass
    
    def read_fixes(self, path: str) -> FixCollection:
        """
        Parses a CSV file and constructs a collection of fixes.

        Args:
            path (str): Path to the FIX_BASE.csv file.

        Returns:
            FixCollection: Indexed enroute fixes.
        """

        collection = FixCollection()

        # We need to specify data types because pandas gets a little confused with those columns
        # csv_rows = pd.read_csv(path, dtype={'ALT_FSS_ID': 'string', 'ALT_FSS_NAME': 'string', 'ALT_TOLL_FREE_NO': 'string', 'ICAO_ID': 'string'})
        csv_rows = pd.read_csv(path)

        for _, row in csv_rows.iterrows():
                fix_id = row['FIX_ID']
                latitude = float(row['LAT_DECIMAL'])
                longitude = float(row['LONG_DECIMAL'])

                fix = Fix(fix_id, latitude, longitude)

                collection.add_fix(fix)

        return collection


if __name__ == '__main__':
    fixes_reader = FixesReader()
    fixes = fixes_reader.read_fixes('../data/FIX_BASE.csv')

    print(fixes.get_by_id('JFK')) # Shouldn't work (Airport)
    print(fixes.get_by_id('LARRI')) # Should work
    print(fixes.get_by_id('J230')) # Shouldn't work (Airway)
    print(fixes.get_by_id('SPI')) # Shouldn't work (VOR Fix)
    print(fixes.get_by_id('TWAIN')) # Should work