import pandas as pd

class Airway:
    def __init__(self, airway_id: str, route: list[str]):
        self.airway_id = airway_id
        self.route = route
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Airway {{ id: {self.airway_id}, route: {self.route} }}'


class AirwayCollection:
    def __init__(self):
        self.airway_id_map = {}

    def add_airway(self, airway: Airway):
        self.airway_id_map[airway.airway_id] = airway
    
    def get_by_id(self, airway_id: str) -> Airway | None:
        return self.airway_id_map.get(airway_id)


class AirwaysReader:
    def __init__(self):
        pass
    
    def read_airways(self, path: str) -> AirwayCollection:
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