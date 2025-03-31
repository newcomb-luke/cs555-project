import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class TrajectoryDataGenerator:
    def __init__(self):
        pass


class TrajectoryDataset(Dataset):
    def __init__(self, json_path: str, context_window_size: int=10):
        self.trajectories = []
        self.context_window_size = context_window_size

        self.attr_names = ['lon', 'lat', 'alt', 'spdx', 'spdy', 'spdz']
        self.data_ranges = {
            "lon": { "max": -60.0, "min": -130.0 },  # Sort of surrounds the US
            "lat": { "max": 65.0, "min": 0.0 },
            "alt": { "max": 60000.0, "min": 0.0 },
            "spdx": { "max": 5480.0, "min": -5480.0 },  # The maxium ground speed
            "spdy": { "max": 5480.0, "min": -5480.0 },
            "spdz": { "max": 15000.0, "min": -15000.0 },
        }
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            for entry in data["e"]:
                self.process_entry(entry)
        
        sums = torch.zeros((6,), dtype=torch.float32)
        mins = torch.full((6,), 100000, dtype=torch.float32)
        maxes = torch.full((6,), -100000, dtype=torch.float32)
        num = 0
        
        for traj in self.trajectories:
            sums += torch.sum(traj['e'], dim=0)
            mins = torch.minimum(mins, torch.min(traj['e'], dim=0).values)
            maxes = torch.maximum(maxes, torch.max(traj['e'], dim=0).values)
            num += len(traj['e'])
        
        averages = sums / num

        print(f'Averages: {averages}')
        print(f'Minimums: {mins}')
        print(f'Maximums: {maxes}')

        exit(1)
        
        random.shuffle(self.trajectories)
    
    def scale(self, value: float, attr: str):
        assert type(attr) is str and attr in self.attr_names
        data_range = self.data_ranges[attr]
        scaled = (value - data_range['min']) / (data_range['max'] - data_range['min'])
        return scaled
    
    def unscale(self, value: float, attr: str):
        assert type(attr) is str and attr in self.attr_names
        data_range = self.data_ranges[attr]
        unscaled = value * (data_range['max'] - data_range['min']) + data_range['min']
        return unscaled

    def coords_in_bounds(self, coords: list[float]) -> bool:
        # Check if we are in bounds for latitude
        return coords[0] <= self.data_ranges['lat']['max'] and coords[0] >= self.data_ranges['lat']['min'] \
                and coords[1] <= self.data_ranges['lon']['max'] and coords[1] >= self.data_ranges['lon']['min']
    
    def process_entry(self, entry):
        start = list(map(float, entry["s"].split(",")))

        if not self.coords_in_bounds(start):
            return

        dest = list(map(float, entry["d"].split(",")))

        if not self.coords_in_bounds(dest):
            return

        # Scale the start and destination
        start_lat = self.scale(start[0], 'lat')
        start_lon = self.scale(start[1], 'lon')
        dest_lat = self.scale(dest[0], 'lat')
        dest_lon = self.scale(dest[1], 'lon')

        start = [start_lat, start_lon]
        dest = [dest_lat, dest_lon]

        trajectory = [list(map(float, p.split(","))) for p in entry["p"]]

        # Scale each of the trajectory points
        points = []
        for p in trajectory:
            # The data files are lon, lat, alt, etc. So lon and lat are swapped
            point_lon = self.scale(p[0], 'lon')
            point_lat = self.scale(p[1], 'lat')
            point_alt = self.scale(p[2], 'alt')
            point_spdx = self.scale(p[3], 'spdx')
            point_spdy = self.scale(p[4], 'spdy')
            point_spdz = self.scale(p[5], 'spdz')

            point = [point_lat, point_lon, point_alt, point_spdx, point_spdy, point_spdz]
            points.append(point)
        
        examples, targets = self.gen_data_from_trajectory(points, self.context_window_size)
        
        start_end = torch.tensor(start + dest).unsqueeze(0).expand(len(examples), self.context_window_size, -1)

        for s_e, e, t in zip(start_end, examples, targets):
            self.trajectories.append({'s_e': s_e, 'e': e, 't': t})
    
    def gen_data_from_trajectory(self, points, context_size: int=10, before_filler: int=-1, after_filler: int=-2):
        window_size = context_size + 1

        before_filler = [before_filler] * 6
        after_filler = [after_filler] * 6

        examples = []
        targets = []

        min_len = min(window_size, len(points) + 1)
        
        for i in range(2, min_len):
            filler_data = [before_filler] * (window_size - i)
            real_data = points[:i]

            data = filler_data + real_data

            example = data[:-1]
            target = data[-1]

            examples.append(example)
            targets.append(target)

        for i in range((len(points) - window_size) + 1):
            real_data = points[i:i+window_size]
            data = real_data

            example = data[:-1]
            target = data[-1]

            examples.append(example)
            targets.append(target)
        
        if window_size <= len(points):
            real_data = points[-(window_size - 1):]
            filler_data = [after_filler]

            data = real_data + filler_data

            example = data[:-1]
            target = data[-1]

            examples.append(example)
            targets.append(target)
        else:
            before_filler_data = [before_filler] * (window_size - min_len)
            real_data = points
            after_filler_data = [after_filler]

            data = before_filler_data + real_data + after_filler_data

            example = data[:-1]
            target = data[-1]

            examples.append(example)
            targets.append(target)
        
        return torch.tensor(examples), torch.tensor(targets)
    
    def collate(self, batch):
        start_end = torch.stack([b[0] for b in batch])
        example = torch.stack([b[1] for b in batch])
        target = torch.stack([b[2] for b in batch])
        return start_end, example, target
    
    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]["s_e"], self.trajectories[idx]["e"], self.trajectories[idx]["t"]


if __name__ == '__main__':
    data = {'e': [
    {
        's': "1.0,2.0",
        'd': "3.0,4.0",
        'p': [
            "31.0,2.0,3.0,4.0,5.0,6.0",
            "32.0,2.0,3.0,4.0,5.0,6.0",
            "33.0,2.0,3.0,4.0,5.0,6.0",
            "34.0,2.0,3.0,4.0,5.0,6.0",
            "35.0,2.0,3.0,4.0,5.0,6.0",
            "36.0,2.0,3.0,4.0,5.0,6.0",
            "37.0,2.0,3.0,4.0,5.0,6.0",
        ]
    },
    {
        's': "54.0,2.0",
        'd': "55.0,4.0",
        'p': [
            "41.0,2.0,3.0,4.0,5.0,6.0",
            "42.0,2.0,3.0,4.0,5.0,6.0",
            "43.0,2.0,3.0,4.0,5.0,6.0",
            "44.0,2.0,3.0,4.0,5.0,6.0",
            "45.0,2.0,3.0,4.0,5.0,6.0",
            "46.0,2.0,3.0,4.0,5.0,6.0",
            "47.0,2.0,3.0,4.0,5.0,6.0",
            "48.0,2.0,3.0,4.0,5.0,6.0",
            "49.0,2.0,3.0,4.0,5.0,6.0",
            "50.0,2.0,3.0,4.0,5.0,6.0",
            "51.0,2.0,3.0,4.0,5.0,6.0",
        ]
    },
    {
        's': "1.0,2.0",
        'd': "3.0,4.0",
        'p': [
            "1.0,2.0,3.0,4.0,5.0,6.0",
            "2.0,2.0,3.0,4.0,5.0,6.0",
            "3.0,2.0,3.0,4.0,5.0,6.0",
            "4.0,2.0,3.0,4.0,5.0,6.0",
            "5.0,2.0,3.0,4.0,5.0,6.0",
            "6.0,2.0,3.0,4.0,5.0,6.0",
            "7.0,2.0,3.0,4.0,5.0,6.0",
            "8.0,2.0,3.0,4.0,5.0,6.0",
            "9.0,2.0,3.0,4.0,5.0,6.0",
            "10.0,2.0,3.0,4.0,5.0,6.0",
            "11.0,2.0,3.0,4.0,5.0,6.0",
            "12.0,2.0,3.0,4.0,5.0,6.0",
            "13.0,2.0,3.0,4.0,5.0,6.0",
            "14.0,2.0,3.0,4.0,5.0,6.0",
            "15.0,2.0,3.0,4.0,5.0,6.0",
            "16.0,2.0,3.0,4.0,5.0,6.0",
            "17.0,2.0,3.0,4.0,5.0,6.0",
        ]
    }, {
        's': "5.0,6.0",
        'd': "7.0,8.0",
        'p': [
            "11.0,2.0,3.0,4.0,5.0,6.0",
            "12.0,2.0,3.0,4.0,5.0,6.0",
            "13.0,2.0,3.0,4.0,5.0,6.0",
            "14.0,2.0,3.0,4.0,5.0,6.0",
            "15.0,2.0,3.0,4.0,5.0,6.0",
            "16.0,2.0,3.0,4.0,5.0,6.0",
            "17.0,2.0,3.0,4.0,5.0,6.0",
            "18.0,2.0,3.0,4.0,5.0,6.0",
            "19.0,2.0,3.0,4.0,5.0,6.0",
            "20.0,2.0,3.0,4.0,5.0,6.0",
            "21.0,2.0,3.0,4.0,5.0,6.0",
            "22.0,2.0,3.0,4.0,5.0,6.0",
            "23.0,2.0,3.0,4.0,5.0,6.0",
            "24.0,2.0,3.0,4.0,5.0,6.0",
            "25.0,2.0,3.0,4.0,5.0,6.0",
            "26.0,2.0,3.0,4.0,5.0,6.0",
            "27.0,2.0,3.0,4.0,5.0,6.0",
        ]
    }
    ]}

    with open('sample.json', 'w') as f:
        json.dump(data, f)

    loader = TrajectoryDataset('sample.json')
    

