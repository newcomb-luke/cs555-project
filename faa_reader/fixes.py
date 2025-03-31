import pandas as pd

class Fix:
    def __init__(self, fix_id: str, latitude: float, longitude: float):
        self.fix_id = fix_id
        self.latitude = latitude
        self.longitude = longitude
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Fix {{ id: {self.fix_id}, lat: {self.latitude}, lon: {self.longitude} }}'


class FixCollection:
    def __init__(self):
        self.fix_id_map = {}

    def add_fix(self, fix: Fix):
        self.fix_id_map[fix.fix_id] = fix
    
    def get_by_id(self, fix_id: str) -> Fix | None:
        return self.fix_id_map.get(fix_id)


class FixesReader:
    def __init__(self):
        pass
    
    def read_fixes(self, path: str) -> FixCollection:
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