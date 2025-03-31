import os
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
    model = TrajectoryTransformer(device, max_seq_len=max_seq_len, d_model=64).to(device)
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


def generate_trajectory(model, device, start_lat, start_lon, end_lat, end_lon, initial_point, seq_len):
    """
    Generates a trajectory given start/end points and an initial trajectory state.

    Args:
        model: Trained TrajectoryTransformer model.
        device: CUDA or CPU.
        start_lat, start_lon: Starting latitude and longitude.
        end_lat, end_lon: Destination latitude and longitude.
        initial_point: Initial [altitude, speed_x, speed_y, speed_z] at start.
        seq_len: Number of steps to predict.

    Returns:
        Predicted trajectory as a numpy array.
    """

    # Define the same scaling ranges as in TrajectoryDataset
    data_ranges = {
        "lon": { "max": -60.0, "min": -130.0 },  # Sort of surrounds the US
        "lat": { "max": 50.0, "min": 25.0 },
        "alt": { "max": 50000.0, "min": 0.0 },
        "spdx": { "max": 685.0, "min": -685.0 },  # The maxium ground speed
        "spdy": { "max": 685.0, "min": -685.0 },
        "spdz": { "max": 3000.0, "min": -3000.0 },
    }

    # Scale input values
    start_lat = scale(start_lat, 'lat', data_ranges)
    start_lon = scale(start_lon, 'lon', data_ranges)
    end_lat = scale(end_lat, 'lat', data_ranges)
    end_lon = scale(end_lon, 'lon', data_ranges)
    initial_point = [scale(initial_point[i], attr, data_ranges) for i, attr in enumerate(["alt", "spdx", "spdy", "spdz"])]

    # Initialize tensors
    start_end = torch.tensor([[start_lat, start_lon, end_lat, end_lon]] * seq_len, dtype=torch.float32).to(device)

    before_filler = -1

    trajectory = torch.full((seq_len, 6), before_filler, dtype=torch.float32).to(device)

    # Set initial step with start position
    trajectory[-1] = torch.tensor([start_lat, start_lon] + initial_point, dtype=torch.float32).to(device)

    print(f'Start: {trajectory[-1]}')

    output_trajectory = [trajectory[-1]]

    total_sequence_len = 200

    model.eval()
    with torch.no_grad():
        for t in range(1, seq_len):
            input_seq = trajectory.unsqueeze(0)  # (1, t, 6)

            start_end_seq = start_end.unsqueeze(0)  # (1, t, 4)

            # Generate padding mask: 1 = ignore, 0 = valid
            tgt_padding_mask = torch.ones((seq_len,), dtype=torch.bool).to(device)  # (seq_len,)
            tgt_padding_mask[-t:] = False  # Mark real data as valid
            tgt_padding_mask = tgt_padding_mask.unsqueeze(0)  # (batch_size, seq_len)

            prediction = model(start_end_seq, input_seq, tgt_padding_mask=tgt_padding_mask)

            trajectory[t] = prediction
            output_trajectory.append(prediction)
        
        for t in range(total_sequence_len - seq_len):
            # Shift it all over by 1 to the left
            trajectory = trajectory[1:, :]
            # We don't need to do anything to start_end

            # Pad with zeros at the end
            padding = torch.zeros((1, 6))
            trajectory = torch.cat([trajectory, padding])

            input_seq = trajectory.unsqueeze(0)  # (1, t, 6)
            start_end_seq = start_end.unsqueeze(0)  # (1, t, 4)

            tgt_padding_mask = torch.zeros((seq_len,), dtype=torch.bool).to(device)  # (seq_len,)
            tgt_padding_mask = tgt_padding_mask.unsqueeze(0)  # (batch_size, seq_len)

            prediction = model(start_end_seq, input_seq, tgt_padding_mask=tgt_padding_mask)

            print(f'{prediction}')
            
            trajectory[-1] = prediction
            output_trajectory.append(prediction)

    predicted_trajectory = []

    for p in output_trajectory:
        unscaled_point = [unscale(p[i], attr, data_ranges) for i, attr in enumerate(["lat", "lon", "alt", "spdx", "spdy", "spdz"])]
        predicted_trajectory.append(unscaled_point)

    print(f'{len(predicted_trajectory)}')

    return predicted_trajectory

def main():
    parser = argparse.ArgumentParser(
        description="Generate a trajectory prediction using a trained Transformer model."
    )
    parser.add_argument('-m', '--model', required=True, help="Path to the trained model checkpoint.")
    parser.add_argument('-s', '--start', nargs=2, type=float, required=True, help="Start latitude and longitude.")
    parser.add_argument('-e', '--end', nargs=2, type=float, required=True, help="End latitude and longitude.")
    parser.add_argument('-i', '--initial', nargs=4, type=float, required=True, help="Initial altitude, speed_x, speed_y, speed_z.")
    parser.add_argument('-o', '--output', default='./predicted_trajectory.npy', help="File to save the predicted trajectory.")
    parser.add_argument('-c', '--context_size', type=int, default=10, help="Size of the context window of the model")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = load_model(device, args.model, args.context_size)

    # Generate trajectory
    predicted_trajectory = generate_trajectory(
        model, device,
        start_lat=args.start[0], start_lon=args.start[1],
        end_lat=args.end[0], end_lon=args.end[1],
        initial_point=args.initial,
        seq_len=args.context_size
    )

    # Save trajectory
    # np.save(args.output, predicted_trajectory)
    # print(f"Predicted trajectory saved to {args.output}")

    x_positions = [p[1] for p in predicted_trajectory]
    y_positions = [p[0] for p in predicted_trajectory]

    time_indices = np.arange(len(predicted_trajectory))

    plt.scatter(x_positions, y_positions, c=time_indices, cmap='copper', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
