import os
import pickle

def verify_batch_consistency(dataloader, num_batches=5, save_path=None):
    """Records the first few batches from a dataloader to check for consistency.

    Args:
        dataloader: The DataLoader instance to check.
        num_batches (int): The number of batches to record.
        save_path (str, optional): Path to save the recorded batch data (pickle file). Defaults to None.

    Returns:
        list: A list of dictionaries, each containing info about a recorded batch.
    """
    batch_data = []
    print(f"Verifying batch consistency for {num_batches} batches...")
    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            # Extract basic info and stats - ensure batch structure matches expected
            # Assuming batch format: (data1, data2, targets, label1, label2)
            if not isinstance(batch, (list, tuple)) or len(batch) < 5:
                 print(f"Warning: Batch {i} has unexpected format. Skipping detailed info.")
                 batch_info = { 'batch_idx': i, 'error': 'Unexpected format' }
            else:
                 batch_info = {
                     'batch_idx': i,
                     'data1_shape': batch[0].shape,
                     'data2_shape': batch[1].shape,
                     'targets': batch[2].tolist(),
                     'labels1': batch[3].tolist(),
                     'labels2': batch[4].tolist(),
                     'data1_sum': batch[0].sum().item(),
                     'data2_sum': batch[1].sum().item(),
                     'data1_std': batch[0].std().item(),
                     'data2_std': batch[1].std().item()
                 }
            batch_data.append(batch_info)
            # print(f"  Recorded batch {i} info.") # Optional verbose logging
    except Exception as e:
        print(f"Error during batch verification: {e}")
        # Optionally add error info to batch_data
        batch_data.append({'batch_idx': i, 'error': str(e)})

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(batch_data, f)
            print(f"Batch consistency data saved to: {save_path}")
        except Exception as e:
            print(f"Error saving batch consistency data to {save_path}: {e}")
            
    print("Batch consistency verification finished.")
    return batch_data
