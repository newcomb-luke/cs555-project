#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: Torch Dataset to generate trajectory training data from serialized flight JSON
#===============================================================================================

import json
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset class for processing scaled trajectory data from JSON training files.
    Each data point includes:
        - start/destination pair
        - scaled waypoints
        - sequence of context trajectory points
        - target next point
    """

    def __init__(self, json_path: str, context_window_size: int=10):
        self.trajectories = []
        self.context_window_size = context_window_size

        self.attr_names = ['lon', 'lat', 'alt', 'spdx', 'spdy', 'spdz']
        self.data_ranges = {
            "lon": { "max": -60.0, "min": -130.0 },  # Sort of surrounds the US
#            "lon": { "max": -60.0, "min": -86.854306 }, # Cutoff sort of at Nashville, TN
            "lat": { "max": 65.0, "min": 0.0 },
            "alt": { "max": 60000.0, "min": 0.0 },
            "spdx": { "max": 5480.0, "min": -5480.0 },  # The maxium ground speed
            "spdy": { "max": 5480.0, "min": -5480.0 },
            "spdz": { "max": 15000.0, "min": -15000.0 },
        }
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            for entry in data["e"]:
                try:
                    self.process_entry(entry)
                except ValueError:
                    pass

        # Used for data analysis
        
        # sums = torch.zeros((6,), dtype=torch.float32)
        # mins = torch.full((6,), 100000, dtype=torch.float32)
        # maxes = torch.full((6,), -100000, dtype=torch.float32)
        # num = 0
        
        # for traj in self.trajectories:
        #     sums += torch.sum(traj['e'], dim=0)
        #     mins = torch.minimum(mins, torch.min(traj['e'], dim=0).values)
        #     maxes = torch.maximum(maxes, torch.max(traj['e'], dim=0).values)
        #     num += len(traj['e'])

        # averages = sums / num
        # print(f'Averages: {averages}')
        # print(f'Minimums: {mins}')
        # print(f'Maximums: {maxes}')

        # exit(1)
        
        # random.shuffle(self.trajectories)
    
    def scale(self, value: float, attr: str):
        """
        Scales a raw value into [0, 1] range using predefined min/max.
        """

        assert type(attr) is str and attr in self.attr_names
        data_range = self.data_ranges[attr]
        scaled = (value - data_range['min']) / (data_range['max'] - data_range['min'])
        return scaled
    
    def unscale(self, value: float, attr: str):
        """
        Unscales a normalized value back to real-world coordinates.
        """

        assert type(attr) is str and attr in self.attr_names
        data_range = self.data_ranges[attr]
        unscaled = value * (data_range['max'] - data_range['min']) + data_range['min']
        return unscaled

    def coords_in_bounds(self, coords: list[float]) -> bool:
        """
        Checks if a latitude/longitude point is inside predefined bounds.
        """

        # Check if we are in bounds for latitude
        return coords[0] <= self.data_ranges['lat']['max'] and coords[0] >= self.data_ranges['lat']['min'] \
                and coords[1] <= self.data_ranges['lon']['max'] and coords[1] >= self.data_ranges['lon']['min']
    
    def process_entry(self, entry):
        """
        Converts a JSON entry into start/dest vectors, waypoints, and trajectory sequences.
        Adds many samples to the dataset by sliding a context window.
        """

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

        waypoint_filler = -2
        waypoint_context_size = 50

        # Scale the waypoints
        waypoints = [list(map(float, wp.split(","))) for wp in entry["w"]]
        wp_scaled = torch.tensor([torch.tensor([self.scale(wp[0], 'lat'), self.scale(wp[1], 'lon')]) for wp in waypoints])
        num_waypoints = wp_scaled.size(0)

        if num_waypoints < waypoint_context_size:
            waypoints_padding = torch.full((waypoint_context_size - num_waypoints, 2), waypoint_filler, dtype=wp_scaled.dtype)
            wp_scaled = torch.cat([wp_scaled, waypoints_padding], dim=0)

        trajectory = [list(map(float, p.split(","))) for p in entry["p"]]

        # Scale each of the trajectory points
        points = []
        for p in trajectory:
            point_lat = self.scale(p[0], 'lat')
            point_lon = self.scale(p[1], 'lon')
            point_alt = self.scale(p[2], 'alt')
            point_spdx = self.scale(p[3], 'spdx')
            point_spdy = self.scale(p[4], 'spdy')
            point_spdz = self.scale(p[5], 'spdz')

            point = [point_lat, point_lon, point_alt, point_spdx, point_spdy, point_spdz]
            points.append(point)
        
        start_end = torch.tensor([start, dest])
        examples, targets = self.gen_data_from_trajectory(points, self.context_window_size)

        for e, t in zip(examples, targets):
            self.trajectories.append({'s_e': start_end, 'w': wp_scaled, 'e': e, 't': t})
    
    def gen_data_from_trajectory(self, points, context_size: int=10, before_filler: int=-1, after_filler: int=-2):
        """
        Generates fixed-size sliding windows from a trajectory.

        Returns:
            tuple[Tensor, Tensor]: input sequences and targets
        """

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
        """
        Collate function for DataLoader batching.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: batched inputs.
        """

        start_end = torch.stack([b[0] for b in batch])
        waypoints = torch.stack([b[1] for b in batch])
        example = torch.stack([b[2] for b in batch])
        target = torch.stack([b[3] for b in batch])
        return start_end, waypoints, example, target
    
    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]["s_e"], self.trajectories[idx]['w'], self.trajectories[idx]["e"], self.trajectories[idx]["t"]