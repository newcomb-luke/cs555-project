#===============================================================================================
# Project: Predicting Commercial Flight Trajectories Using Transformers for CS 555
# Author(s): Luke, Kayla
# Description: Main training script for transformer model using processed trajectory JSON data
#===============================================================================================

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import TrajectoryDataset
from model import TrajectoryTransformer

def main():
    parser = argparse.ArgumentParser(
        prog='Custom Transformer Trainer',
        description='Trains a transformer-based flight prediction model'
    )
    parser.add_argument('-d', '--data', default='../data/', help='The directory containing the data')
    parser.add_argument('-o', '--output', default='./output/', help='The directory for model output')
    parser.add_argument('-e', '--epochs', default=100, help='The number of epochs to train for')

    args = parser.parse_args()

    epochs = args.epochs

    try:
        epochs = int(epochs)
    except ValueError:
        print(f'Provided number of epochs `{epochs}` is not an integer')
        exit(1)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args.data

    train_path = os.path.join(data_dir, 'train.json')
    test_path = os.path.join(data_dir, 'test.json')
    validation_path = os.path.join(data_dir, 'dev.json')

    train_dataset = TrajectoryDataset(train_path, context_window_size=10)
    test_dataset = TrajectoryDataset(test_path, context_window_size=10)
    validation_dataset = TrajectoryDataset(validation_path)

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=train_dataset.collate)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True, collate_fn=test_dataset.collate)

    print('Dataset loaded')

    output_dir = args.output
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)

    model = TrajectoryTransformer(device, max_seq_len=10, d_model=64).to(device)

    mse = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses_file_path = os.path.join(output_dir, 'losses.txt')
    with open(losses_file_path, 'a') as f:
        f.write(f'epoch,avg_loss\n')

    num_epochs = epochs
    for epoch in range(num_epochs):
        # Train for one epoch
        loss_val = train_one_epoch(device, model, optimizer, mse, train_dataloader, epoch)

        # Output the current training status
        print(f"Epoch {epoch+1}/{num_epochs}, Average loss: {loss_val:.4f}")

        # Save the current checkpoint after each epoch
        checkpoint_name = f'checkpoint-epoch-{epoch}.pth'
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        save_model(model, optimizer, epoch, checkpoint_path)

        with open(losses_file_path, 'a') as f:
            f.write(f'{epoch},{loss_val}\n')
    

    # Save the final model
    model_name = f'final-checkpoint.pth'
    model_path = os.path.join(output_dir, model_name)
    save_model(model, optimizer, num_epochs - 1, model_path)


def save_model(model, optimizer, epoch, model_path):
    """
    Saves the model and optimizer state to a file.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)


def load_model(device, model_path):
    """
    Loads the model and optimizer state from a file.
    """
    model = TrajectoryTransformer(device).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def calculate_spatial_loss(device, target: torch.Tensor, predicted: torch.Tensor, mse, is_training=True) -> torch.Tensor:
    """
    Computes spatial loss with weighted emphasis on the final point.

    Args:
        device: torch device
        target: (batch, seq_len, 6) target tensor
        predicted: (batch, seq_len, 6) predicted tensor
        mse: MSE loss function
        is_training: whether training or inference

    Returns:
        torch.Tensor: computed loss
    """

    context_size = target.shape[1]

    if is_training:
        # We want to weight all parts of the sequence evenly
        weights = torch.ones(context_size, dtype=torch.float32, device=device)
    else:
        weights = torch.zeros(context_size, dtype=torch.float32, device=device)
    
    # Training or not, the last point has a weight of 1.0
    weights[-1] = 1.0

    weighted_loss = torch.tensor(0.0, device=device)

    for i in range(context_size):
        weighted_loss += weights[i] * mse(predicted[:, i], target[:, i, :])
    
    if is_training:
        weighted_loss /= torch.tensor(context_size, dtype=torch.float32, device=device)
    
    return weighted_loss


def haversine_distance(lat1, lon1, lat2, lon2, radius_km=6371.0):
    """
    lat/lon must be in degrees, shape: (batch, seq_len) or (N,)
    """
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return radius_km * c  # shape: (batch, seq_len)


def batch_unscale(tensor: torch.Tensor, data_ranges: dict) -> torch.Tensor:
    """
    Unscales an entire batch of normalized trajectories using provided data ranges.

    Args:
        tensor: shape (batch, seq_len, features)
        data_ranges: dict of min/max for each feature

    Returns:
        Unnormalized tensor, same shape
    """
    unscaled = torch.empty_like(tensor)
    keys = ['lat', 'lon', 'alt', 'spdx', 'spdy', 'spdz']
    for i, key in enumerate(keys):
        min_val = data_ranges[key]['min']
        max_val = data_ranges[key]['max']
        unscaled[..., i] = tensor[..., i] * (max_val - min_val) + min_val
    return unscaled


def trajectory_haversine_loss(predicted: torch.Tensor, target: torch.Tensor, data_ranges: dict, sentinel_val=-1.5) -> torch.Tensor:
    """
    Computes mean haversine distance between predicted and target lat/lon at each timestep,
    skipping padded/sentinel points.

    Args:
        predicted: (batch, seq_len, 6) - normalized
        target: (batch, seq_len, 6) - normalized
        data_ranges: dict with 'lat' and 'lon' → (min, max)
        sentinel_val: padding marker in normalized space

    Returns:
        Scalar loss in kilometers
    """
    device = predicted.device

    # Identify valid (non-sentinel) positions
    valid_mask = (target[..., 0:2] > sentinel_val).any(dim=-1)  # (batch, seq_len)

    # Extract lat/lon and unnormalize
    pred_unscaled = batch_unscale(predicted, data_ranges)
    tgt_unscaled = batch_unscale(target, data_ranges)

    lat_pred = pred_unscaled[..., 0]
    lon_pred = pred_unscaled[..., 1]
    lat_tgt = tgt_unscaled[..., 0]
    lon_tgt = tgt_unscaled[..., 1]

    # Compute distance for all positions
    distances = haversine_distance(lat_pred, lon_pred, lat_tgt, lon_tgt)  # (batch, seq_len)

    # Mask out invalid positions
    distances = distances * valid_mask.float()

    # Average over valid points
    total_valid = valid_mask.sum()
    if total_valid == 0:
        return torch.tensor(0.0, device=device)

    return distances.sum() / total_valid


def unscale(tensor: torch.Tensor, key: str, data_ranges: dict) -> torch.Tensor:
    """
    Unscales a normalized tensor value to its original range using min-max scaling.

    Args:
        tensor: normalized tensor (e.g. between 0 and 1 or -1.5 to 1.5)
        key: the feature key, e.g., 'lat', 'lon', etc.
        data_ranges: dictionary with keys → { "min": x, "max": y }

    Returns:
        Unnormalized tensor (same shape as input)
    """
    min_val = data_ranges[key]["min"]
    max_val = data_ranges[key]["max"]

    return tensor * (max_val - min_val) + min_val


def train_one_epoch(device, model, optimizer, mse, dataloader, epoch: int):
    """
    Trains the model for one epoch.
    """

    model.train()

    total_loss = 0
    data_len = len(dataloader)

    data_ranges = {
        "lon": { "max": -60.0, "min": -130.0 },  # Sort of surrounds the US
        "lat": { "max": 65.0, "min": 0.0 },
        "alt": { "max": 60000.0, "min": 0.0 },
        "spdx": { "max": 5480.0, "min": -5480.0 },  # The maxium ground speed
        "spdy": { "max": 5480.0, "min": -5480.0 },
        "spdz": { "max": 15000.0, "min": -15000.0 },
    }

    for index, (start_end, waypoints, trajectory, target) in enumerate(dataloader):
        # start_end.shape = (1024, 100, 4)
        # trajectory.shape = (1024, 100, 6)
        #               batch, sequence, attributes
        start_end, waypoints, trajectory, target = start_end.to(device), waypoints.to(device), trajectory.to(device), target.to(device)

        start = start_end[:, 0].unsqueeze(1)
        end = start_end[:, 1].unsqueeze(1)

        all_waypoints = torch.cat((start, waypoints, end), dim=1)

        optimizer.zero_grad()

        input_seq = trajectory

        target_seq = torch.concat((input_seq[:, 1:, :], target.unsqueeze(1)), dim=1)

        # Generate padding mask: (batch, seq_len - 1) where 1 = ignore, 0 = valid
        tgt_padding_mask = (input_seq.sum(dim=-1) < 0).bool().to(device)  # Identify padded positions

        # output shape: (1024, 99, 6)
        prediction = model(all_waypoints, input_seq, tgt_padding_mask=tgt_padding_mask)

        spatial_loss = calculate_spatial_loss(device, target_seq, prediction, mse, is_training=True)
        # haversine_loss = trajectory_haversine_loss(prediction, target_seq, data_ranges)

        loss_backward = spatial_loss # + haversine_loss

        loss_backward.backward()
        optimizer.step()

        total_loss += loss_backward.item()

        percentage = ((index + 1) / data_len) * 100.0
        print(f'Epoch {epoch}:   progress={percentage:.1f}%, batch={index + 1}/{data_len}, loss={loss_backward.item():.4f}')

    avg_loss = total_loss / data_len
    return avg_loss


if __name__ == "__main__":
    main()
