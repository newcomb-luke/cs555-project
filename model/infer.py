import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from model import TrajectoryTransformer
import matplotlib.pyplot as plt


def load_model(device, model_path, max_seq_len, d_model=64):
    """
    Loads a trained model from a checkpoint.
    """
    model = TrajectoryTransformer(device, max_seq_len=max_seq_len, d_model=d_model).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def scale(value, attr, data_ranges):
    """Scales a value between 0 and 1 based on dataset ranges."""
    return (value - data_ranges[attr]['min']) / (data_ranges[attr]['max'] - data_ranges[attr]['min'])


def unscale(value, attr, data_ranges):
    """Unscales a value from [0,1] back to original range."""
    return value * (data_ranges[attr]['max'] - data_ranges[attr]['min']) + data_ranges[attr]['min']


def generate_trajectory(model, device, sample, seq_len):
    """
    Generates a trajectory given start/end points and an initial trajectory state.

    Args:
        model: Trained TrajectoryTransformer model.
        device: CUDA or CPU.
        sample: The data for the first point / waypoints
        seq_len: Number of steps to predict.

    Returns:
        Predicted trajectory as a numpy array.
    """

    # Define the same scaling ranges as in TrajectoryDataset
    data_ranges = {
        "lon": { "max": -60.0, "min": -130.0 },  # Sort of surrounds the US
        "lat": { "max": 65.0, "min": 0.0 },
        "alt": { "max": 60000.0, "min": 0.0 },
        "spdx": { "max": 5480.0, "min": -5480.0 },  # The maxium ground speed
        "spdy": { "max": 5480.0, "min": -5480.0 },
        "spdz": { "max": 15000.0, "min": -15000.0 },
    }

    start = [float(x) for x in sample['s'].split(',')]
    end = [float(x) for x in sample['d'].split(',')]
    initial_point = [float(x) for x in sample['p'][0].split(',')][2:]
    waypoint_strings = [s.split(',') for s in sample['w']]
    waypoints = [[float(x) for x in p] for p in waypoint_strings]

    # Scale input values
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
    print(f'start_wp shape: {start_wp.shape}')
    print(f'end_wp shape: {end_wp.shape}')
    print(f'scaled_waypoints shape: {scaled_waypoints.shape}')
    waypoints = torch.cat((start_wp, scaled_waypoints, end_wp), dim=0)

    before_filler = -1

    trajectory = torch.full((seq_len, 6), before_filler, dtype=torch.float32).to(device)

    # Set initial step with start position
    trajectory[-1] = torch.tensor([start_lat, start_lon] + initial_point, dtype=torch.float32).to(device)

    print(f'Start: {trajectory[-1]}')
    print(f'Trajectory shape: {trajectory.shape}')
    print(f'Waypoints shape: {waypoints.shape}')

    output_trajectory = [trajectory[-1]]

    total_sequence_len = 100

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
        
        # for t in range(total_sequence_len - seq_len):
        #     # Shift it all over by 1 to the left
        #     trajectory = trajectory[1:, :]
        #     # We don't need to do anything to start_end

        #     # Pad with zeros at the end
        #     padding = torch.zeros((1, 6))
        #     trajectory = torch.cat([trajectory, padding])

        #     input_seq = trajectory.unsqueeze(0)  # (1, t, 6)
        #     input_waypoints = waypoints.unsqueeze(0) # (1, 52, 2)

        #     tgt_padding_mask = torch.zeros((seq_len,), dtype=torch.bool).to(device)  # (seq_len,)
        #     tgt_padding_mask = tgt_padding_mask.unsqueeze(0)  # (batch_size, seq_len)

        #     prediction = model(input_waypoints, input_seq, tgt_padding_mask=tgt_padding_mask)
        #     prediction = prediction[0][-1]

        #     print(f'{prediction}')
        #     
        #     trajectory[-1] = prediction
        #     output_trajectory.append(prediction)

    predicted_trajectory = []

    for p in output_trajectory:
        unscaled_point = [unscale(p[i], attr, data_ranges) for i, attr in enumerate(["lat", "lon", "alt", "spdx", "spdy", "spdz"])]
        print(f'Unscaled predicted point: {unscaled_point}')
        predicted_trajectory.append(unscaled_point)

    print(f'{len(predicted_trajectory)}')

    return predicted_trajectory

def main():
    parser = argparse.ArgumentParser(
        description="Generate a trajectory prediction using a trained Transformer model."
    )
    parser.add_argument('-m', '--model', required=True, help="Path to the trained model checkpoint.")
    parser.add_argument('-d', '--data', required=True, help="Path to the data to get the example from, .json file")
    parser.add_argument('-s', '--sample', required=True, type=int, help="Path to the data to get the example from, .json file")
    parser.add_argument('-o', '--output', default='./predicted_trajectory.npy', help="File to save the predicted trajectory.")
    parser.add_argument('-c', '--context_size', type=int, default=10, help="Size of the context window of the model")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = load_model(device, args.model, args.context_size)

    # Load the data file
    with open(args.data, 'r') as f:
        data = json.load(f)
    
    # Get the sample
    sample = data['e'][args.sample]

    print(f'Sample: {sample}')

    # Generate trajectory
    predicted_trajectory = generate_trajectory(
        model, device,
        sample,
        seq_len=args.context_size
    )

    # Save trajectory
    # np.save(args.output, predicted_trajectory)
    # print(f"Predicted trajectory saved to {args.output}")

    x_positions = [p[1] for p in predicted_trajectory]
    y_positions = [p[0] for p in predicted_trajectory]

    time_indices = np.arange(len(predicted_trajectory))

    start = [float(x) for x in sample['s'].split(',')]
    end = [float(x) for x in sample['d'].split(',')]
    start_x = start[1]
    start_y = start[0]
    end_x = end[1]
    end_y = end[0]

    real_points = [[float(x) for x in p] for p in [s.split(',')[:2] for s in sample['p']]]
    real_points_x = [p[1] for p in real_points]
    real_points_y = [p[0] for p in real_points]

    for real_point in real_points:
        print(f'Real point: {torch.tensor(real_point)}')

    # waypoint_positions = [[float(x) for x in p] for p in [s.split(',') for s in sample['w']]]
    # waypoints_x = [p[1] for p in waypoint_positions]
    # waypoints_y = [p[0] for p in waypoint_positions]

    plt.scatter(x_positions, y_positions, c=time_indices, cmap='copper', alpha=0.7)
    # plt.scatter(waypoints_x, waypoints_y)
    plt.scatter(real_points_x, real_points_y)
    plt.scatter([start_x], [start_y])
    plt.scatter([end_x], [end_y])
    plt.show()

if __name__ == "__main__":
    main()
