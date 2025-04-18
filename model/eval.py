#==================================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): 
# Description: A script used for evaluating the performance of the trajectory prediction model
#==================================================================================================

import json
import argparse
import torch
import numpy as np
from model import TrajectoryTransformer
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


def load_model(device, model_path, max_seq_len, d_model=256):
    """
    Loads a saved transformer model from checkpoint.

    Args:
        device: torch device (e.g., 'cpu' or 'cuda')
        model_path: path to the saved .pth file
        max_seq_len: context window length
        d_model: embedding dimension (default 256)

    Returns:
        Loaded TrajectoryTransformer model in eval mode
    """

    model = TrajectoryTransformer(device, max_seq_len=max_seq_len, d_model=d_model).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def scale(value, attr, data_ranges):
    """
    Scales a raw value using min-max scaling for the specified attribute.
    """
    return (value - data_ranges[attr]['min']) / (data_ranges[attr]['max'] - data_ranges[attr]['min'])


def unscale(value, attr, data_ranges):
    """
    Reverses min-max scaling to recover the original value.
    """
    return value * (data_ranges[attr]['max'] - data_ranges[attr]['min']) + data_ranges[attr]['min']


def compute_ade(predicted, target):
    """
    Computes Average Displacement Error (ADE).

    Args:
        predicted: list of predicted (lat, lon) points
        target: list of ground truth (lat, lon) points

    Returns:
        Mean Euclidean distance over all time steps
    """

    return np.mean([euclidean(p, t) for p, t in zip(predicted, target)])


def compute_fde(predicted, target):
    """
    Computes Final Displacement Error (FDE).

    Args:
        predicted: list of predicted (lat, lon) points
        target: list of ground truth (lat, lon) points

    Returns:
        Euclidean distance between final predicted and target point
    """

    return euclidean(predicted[-1], target[-1])


def evaluate(model, device, sample, seq_len, data_ranges):
    """
    Runs a forward autoregressive prediction pass and returns unscaled trajectory.

    Args:
        model: loaded TrajectoryTransformer model
        device: torch device
        sample: single JSON record with 's', 'd', 'w', 'p' fields
        seq_len: context window length
        data_ranges: dictionary for min-max scaling bounds

    Returns:
        List of unscaled predicted trajectory points (lat, lon, alt, spdx, spdy, spdz)
    """

    start = [float(x) for x in sample['s'].split(',')]
    end = [float(x) for x in sample['d'].split(',')]
    initial_point = [float(x) for x in sample['p'][0].split(',')][2:]
    waypoint_strings = [s.split(',') for s in sample['w']]
    waypoints = [[float(x) for x in p] for p in waypoint_strings]

    start_lat = scale(start[0], 'lat', data_ranges)
    start_lon = scale(start[1], 'lon', data_ranges)
    end_lat = scale(end[0], 'lat', data_ranges)
    end_lon = scale(end[1], 'lon', data_ranges)
    initial_point = [scale(initial_point[i], attr, data_ranges) for i, attr in enumerate(["alt", "spdx", "spdy", "spdz"])]
    scaled_waypoints = []

    for waypoint in waypoints:
        scaled = [scale(waypoint[i], attr, data_ranges) for i, attr in enumerate(['lat', 'lon'])]
        scaled_waypoints.append(scaled)

    scaled_waypoints = torch.tensor(scaled_waypoints)

    waypoint_filler = -2
    waypoint_context_size = 50

    num_waypoints = scaled_waypoints.size(0)

    if num_waypoints < waypoint_context_size:
        waypoints_padding = torch.full((waypoint_context_size - num_waypoints, 2), waypoint_filler, dtype=scaled_waypoints.dtype)
        scaled_waypoints = torch.cat([scaled_waypoints, waypoints_padding], dim=0)

    start_wp = torch.tensor([[start_lat, start_lon]])
    end_wp = torch.tensor([[end_lat, end_lon]])
    waypoints = torch.cat((start_wp, scaled_waypoints, end_wp), dim=0)

    before_filler = -1
    trajectory = torch.full((seq_len, 6), before_filler, dtype=torch.float32).to(device)
    trajectory[-1] = torch.tensor([start_lat, start_lon] + initial_point, dtype=torch.float32).to(device)

    output_trajectory = [trajectory[-1]]
    total_sequence_len = 80

    model.eval()
    with torch.no_grad():
        for t in range(1, seq_len):
            # Gather last t predictions (or 1 known + t-1 predictions)
            context = output_trajectory[-min(t, seq_len):]  # list of tensors

            # Pad with before_filler on the left if needed
            if len(context) < seq_len:
                padding_len = seq_len - len(context)
                padding = [torch.full((6,), before_filler).to(device)] * padding_len
                input_seq = torch.stack(padding + context)  # shape: (seq_len, 6)
            else:
                input_seq = torch.stack(context)  # shape: (seq_len, 6)

            tgt_padding_mask = (input_seq.sum(dim=-1) < 0).unsqueeze(0)
            input_seq = input_seq.unsqueeze(0)  # shape: (1, seq_len, 6)

            prediction = model(waypoints.unsqueeze(0), input_seq, tgt_padding_mask=tgt_padding_mask)
            prediction = prediction[0][-1]

            output_trajectory.append(prediction.clone())

        for t in range(total_sequence_len - seq_len):
                input_seq = trajectory.unsqueeze(0)  # (1, seq_len, 6)
                input_waypoints = waypoints.unsqueeze(0)  # (1, 52, 2)

                tgt_padding_mask = (trajectory.sum(dim=-1) < 0).unsqueeze(0)  # (1, seq_len)

                prediction = model(input_waypoints, input_seq, tgt_padding_mask=tgt_padding_mask)
                next_point = prediction[0, -1]

                trajectory = torch.roll(trajectory, shifts=-1, dims=0)
                trajectory[-1] = next_point

                output_trajectory.append(next_point.clone())

    predicted_trajectory = []

    for p in output_trajectory:
        unscaled_point = [unscale(p[i], attr, data_ranges) for i, attr in enumerate(["lat", "lon", "alt", "spdx", "spdy", "spdz"])]
        print(f'Unscaled predicted point: {unscaled_point}')
        predicted_trajectory.append(unscaled_point)

    print(f'{len(predicted_trajectory)}')

    return predicted_trajectory


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory prediction using a trained Transformer model.")
    parser.add_argument('-m', '--model', required=True, help="Path to the trained model checkpoint.")
    parser.add_argument('-d', '--data', required=True, help="Path to the .json file containing samples.")
    parser.add_argument('-s', '--sample', required=True, type=int, help="Sample index to evaluate.")
    parser.add_argument('-c', '--context_size', type=int, default=10, help="Context window size of the model.")

    args = parser.parse_args()
    device = 'cpu'

    model = load_model(device, args.model, args.context_size)

    with open(args.data, 'r') as f:
        data = json.load(f)

    sample = data['e'][args.sample]

    data_ranges = {
        "lon": { "max": -60.0, "min": -130.0 },
        "lat": { "max": 65.0, "min": 0.0 },
        "alt": { "max": 60000.0, "min": 0.0 },
        "spdx": { "max": 5480.0, "min": -5480.0 },
        "spdy": { "max": 5480.0, "min": -5480.0 },
        "spdz": { "max": 15000.0, "min": -15000.0 },
    }

    # Load ground truth
    real_points = [[float(x) for x in p] for p in [s.split(',')[:2] for s in sample['p']]]

    predicted_trajectory = evaluate(model, device, sample, args.context_size, data_ranges)
    predicted_points = [p[:2] for p in predicted_trajectory[:len(real_points)]]

    ade = compute_ade(predicted_points, real_points)
    fde = compute_fde(predicted_points, real_points)

    print(f"ADE: {ade:.3f}")
    print(f"FDE: {fde:.3f}")

    # Plotting
    pred_x = [p[1] for p in predicted_points]
    pred_y = [p[0] for p in predicted_points]
    real_x = [p[1] for p in real_points]
    real_y = [p[0] for p in real_points]

    plt.plot(real_x, real_y, label='Real', marker='o')
    plt.plot(pred_x, pred_y, label='Predicted', marker='x')
    plt.legend()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Trajectory Prediction vs. Ground Truth")
    plt.show()


if __name__ == "__main__":
    main()
