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
    
    max_level = 1

    model = TrajectoryTransformer(device, max_level, max_seq_len=10, d_model=64).to(device)

    mse = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses_file_path = os.path.join(output_dir, 'losses.txt')
    with open(losses_file_path, 'a') as f:
        f.write(f'epoch,avg_loss\n')

    num_epochs = 100
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


def train_one_epoch(device, model, optimizer, mse, dataloader, epoch: int):
    model.train()

    total_loss = 0
    data_len = len(dataloader)

    for index, (start_end, trajectory, target) in enumerate(dataloader):
        # start_end.shape = (1024, 100, 4)
        # trajectory.shape = (1024, 100, 6)
        #               batch, sequence, attributes
        start_end, trajectory, target = start_end.to(device), trajectory.to(device), target.to(device)

        optimizer.zero_grad()

        input_seq = trajectory

        target_seq = torch.concat((input_seq[:, 1:, :], target.unsqueeze(1)), dim=1)

        # Generate padding mask: (batch, seq_len - 1) where 1 = ignore, 0 = valid
        tgt_padding_mask = (input_seq.sum(dim=-1) < 0).bool().to(device)  # Identify padded positions

        # output shape: (1024, 99, 6)
        prediction = model(start_end, input_seq, tgt_padding_mask=tgt_padding_mask)

        spatial_loss = calculate_spatial_loss(device, target_seq, prediction, mse, is_training=True)

        loss_backward = spatial_loss

        loss_backward.backward()
        optimizer.step()

        total_loss += loss_backward.item()

        percentage = ((index + 1) / data_len) * 100.0
        print(f'Epoch {epoch}:   progress={percentage:.1f}%, batch={index + 1}/{data_len}, loss={loss_backward.item():.4f}')

    avg_loss = total_loss / data_len
    return avg_loss


if __name__ == "__main__":
    main()
