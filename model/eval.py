import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import TrajectoryDataset
from model import TrajectoryTransformer


def load_model(device, model_path, max_seq_len):
    """
    Loads the trained model from a checkpoint.
    """
    model = TrajectoryTransformer(device, max_seq_len=max_seq_len).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on the given test dataset.
    """
    model.eval()
    mse_loss = nn.MSELoss(reduction='mean').to(device)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for start_end, trajectory in dataloader:
            start_end, trajectory = start_end.to(device), trajectory.to(device)

            input_seq = trajectory[:, :-1, :]  # First (seq_len-1) steps
            target_seq = trajectory[:, 1:, :]  # Next (seq_len-1) steps

            # Generate padding mask: (batch, seq_len-1)
            tgt_padding_mask = (input_seq.sum(dim=-1) == 0).bool().to(device)

            # Run model inference
            prediction = model(start_end[:, :-1, :], input_seq, tgt_padding_mask=tgt_padding_mask)

            # Convert prediction back to original scale if needed
            prediction = prediction[-1]  # Last level of prediction
            loss = mse_loss(prediction, trajectory)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Evaluation Complete. MSE Loss: {avg_loss:.6f}")

    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TrajectoryTransformer model on the test dataset."
    )
    parser.add_argument('-m', '--model', required=True, help="Path to the trained model checkpoint.")
    parser.add_argument('-d', '--data', default='../data/', help="Directory containing test data.")
    parser.add_argument('-b', '--batch_size', type=int, default=256, help="Batch size for evaluation.")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 100

    # Load test dataset
    test_path = os.path.join(args.data, 'test.json')
    test_dataset = TrajectoryDataset(test_path, max_len=seq_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate)

    print("Test dataset loaded")

    # Load model
    model = load_model(device, args.model, seq_len)

    # Evaluate model
    evaluate_model(model, test_dataloader, device)


if __name__ == "__main__":
    main()
