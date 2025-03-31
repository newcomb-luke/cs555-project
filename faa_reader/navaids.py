import pandas as pd

class Navaid:
    def __init__(self, navaid_id: str, latitude: float, longitude: float):
        self.navaid_id = navaid_id
        self.latitude = latitude
        self.longitude = longitude
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Navaid {{ id: {self.navaid_id}, lat: {self.latitude}, lon: {self.longitude} }}'


class NavaidCollection:
    def __init__(self):
        self.navaid_id_map = {}

    def add_navaid(self, navaid: Navaid):
        self.navaid_id_map[navaid.navaid_id] = navaid
    
    def get_by_id(self, navaid_id: str) -> Navaid | None:
        return self.navaid_id_map.get(navaid_id)


class NavaidsReader:
    def __init__(self):
        pass
    
    def read_navaids(self, path: str) -> NavaidCollection:
        collection = NavaidCollection()

        # We need to specify data types because pandas gets a little confused with those columns
        csv_rows = pd.read_csv(path)

        for _, row in csv_rows.iterrows():
                navaid_id = row['NAV_ID']
                latitude = float(row['LAT_DECIMAL'])
                longitude = float(row['LONG_DECIMAL'])

                navaid = Navaid(navaid_id, latitude, longitude)

                collection.add_navaid(navaid)

        return collection


if __name__ == '__main__':
    navaids_reader = NavaidsReader()
    navaids = navaids_reader.read_navaids('../data/NAV_BASE.csv')

    print(navaids.get_by_id('AIR'))
    print(navaids.get_by_id('ILC'))
    print(navaids.get_by_id('DSM'))