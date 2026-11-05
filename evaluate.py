import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from celeba_classifier.celeba import CelebA
# from celeba_classifier.celeba_vgg16_train import AttributeNN
from celeba_classifier.celeba import CelebA_ViT as CelebA
from celeba_classifier.celeba_vit_train import AttributeNN
# from celeba_classifier.celeba_mobilenetv2_train import AttributeNN
# from celeba_classifier.celeba_resnet_train import AttributeNN
import numpy as np
import argparse
# def evaluate(network, dataloader, device):

#     for batch, batch_labels in dataloader:
#         batch = batch.to(device)
#         with torch.no_grad():
#             out = network(batch)
#             out = torch.sigmoid(out).cpu().numpy()
def evaluate(network, dataloader, device):
    all_outputs = []
    all_labels = []
    all_preds = []
    
    loss_function = nn.BCEWithLogitsLoss()
    
    for batch, labels in dataloader:
        batch = batch.to(device)
        labels = labels.float().to(device)
        with torch.no_grad():
            output = network.forward(batch)
            loss = loss_function(output, labels)
            preds = (torch.sigmoid(output) > 0.5).type(torch.int)
            
        all_outputs.append(output.cpu())  # Store logits instead of sigmoid
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
    
    # Concatenate all results
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    
    # Calculate overall metrics
    tp = torch.zeros(40)
    tn = torch.zeros(40)
    fp = torch.zeros(40)
    fn = torch.zeros(40)
    
    tp = torch.sum(all_preds + all_labels == 2, axis=0).type(torch.float)
    fp = torch.sum(all_preds - all_labels == 1, axis=0).type(torch.float)
    tn = torch.sum(all_preds + all_labels == 0, axis=0).type(torch.float)
    fn = torch.sum(all_preds - all_labels == -1, axis=0).type(torch.float)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2*(precision*recall)/(precision + recall)
    balanced_acc = .5*(tp/(tp + fn) + tn/(tn + fp))

    avg_accuracy = (tp.sum()+tn.sum())/(tp.sum()+tn.sum()+fp.sum()+fn.sum())
    avg_precision = tp.sum()/(tp.sum() + fp.sum())
    avg_recall = tp.sum()/(tp.sum() + fn.sum())
    avg_f1 = 2*(avg_precision*avg_recall)/(avg_precision + avg_recall)
    avg_balanced_acc = .5*(tp.sum()/(tp.sum() + fn.sum())
                           + tn.sum()/(tn.sum() + fp.sum()))

    return {"acc" : accuracy, "bal_acc" : balanced_acc, "f1" : f1,
            "precision" : precision, "recall" : recall,
            "avg_acc" : avg_accuracy.item(),
            "avg_bal_acc" : avg_balanced_acc.item(), "avg_f1" : avg_f1.item(),
            "avg_precision" : avg_precision.item(),
            "avg_recall" : avg_recall.item(),
            "preds" : all_preds,
            "outputs" : all_outputs,
            "labels" : all_labels}

def calculate_per_image_epr(outputs_before, outputs_after, target_idx, non_target_indices):
    """
    Calculate EPR for each individual image
    outputs_before/after: [N, 40] tensor of logits (raw network outputs)
    """
    # Calculate per-image EPR using raw logits
    delta_target = torch.abs(outputs_after[:, target_idx] - outputs_before[:, target_idx])
    delta_non_targets = torch.abs(outputs_after[:, non_target_indices] - outputs_before[:, non_target_indices])
    delta_non_targets_mean = delta_non_targets.mean(dim=1)  # Mean across non-target attributes for each image
    
    # EPR for each image
    epr_per_image = delta_target / (delta_non_targets_mean + 1e-8)
    
    return epr_per_image.numpy()

def nls_metric(attrs_before, attrs_after, target_idx, non_target_indices):
    delta_target = np.abs(attrs_after[target_idx] - attrs_before[target_idx]).mean()
    delta_non_targets = np.abs(attrs_after[non_target_indices] - attrs_before[non_target_indices]).mean()
    delta_non_targets_mean = delta_non_targets.mean()
    nls = delta_target / (delta_non_targets_mean + 1e-8)
    return nls, delta_target, delta_non_targets_mean

def label_metrics(attrs_before, attrs_after, y_true, target_idx):
    import pdb; pdb.set_trace()
    pred_before = (np.sigmoid(attrs_before) >= 0.5).astype(np.int32)
    pred_after = (attrs_after >= 0.5).astype(np.int32)*2 - 1
    acc_before = (pred_before == y_true).mean(axis=0)
    acc_after = (pred_after == y_true).mean(axis=0)
    acc_target_before = acc_before[target_idx]
    acc_target_after = acc_after[target_idx]
    flip_rate = (pred_before[:, target_idx] != pred_after[:, target_idx]).mean()
    return acc_target_before, acc_target_after, flip_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before_dir', required=True, help='Directory containing original images (before editing)')
    parser.add_argument('--after_dir', required=True, help='Directory containing edited images (after editing)')
    parser.add_argument('--model_path', required=True, help='Path to attribute classification model weights')
    parser.add_argument('--label_file', required=True, help='Path to CelebA-style attribute label file')
    parser.add_argument('--target_concept', required=True, help='Target attribute to evaluate, e.g., Smiling')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_path', default=None, help='Optional: path to save metric results')
    args = parser.parse_args()

    # 1. Load label file and attribute names
    with open(args.label_file, 'r') as f:
        lines = f.readlines()
    attr_names = lines[1].strip().split()
    label_dict = {}
    for line in lines[2:]:
        parts = line.strip().split()
        fname = parts[0]
        attrs = [int(x) for x in parts[1:]]
        label_dict[fname] = attrs
    # import pdb; pdb.set_trace()
    target_idx = attr_names.index(args.target_concept)
    non_target_indices = [i for i in range(len(attr_names)) if i != target_idx]

    # 2. Build CelebA datasets for before and after images
    dataset_before = CelebA(data_folder=args.before_dir, normalize=False)
    dataset_after = CelebA(data_folder=args.after_dir, normalize=False)

    # Ensure both datasets are in same order and filenames match
    assert len(dataset_before) == len(dataset_after)
    img_names_before = [dataset_before.images[i] for i in range(len(dataset_before))]
    img_names_after = [dataset_after.images[i] for i in range(len(dataset_after))]
    # import pdb; pdb.set_trace()
    # assert img_names_before == img_names_after, "Image order mismatch!"

    dataloader_before = DataLoader(dataset_before, batch_size=args.batch_size, num_workers=0)
    dataloader_after = DataLoader(dataset_after, batch_size=args.batch_size, num_workers=0)

    # 3. Build network and load weights
    device = args.device
    network = AttributeNN(40, True)
    network.to(device)
    network.eval()
    network.load_state_dict(torch.load(args.model_path, map_location=device))

    # 4. Forward pass to get attributes for before and after
    metrics_before = evaluate(network, dataloader_before, device)
    metrics_after = evaluate(network, dataloader_after, device)
    
    # Extract outputs and labels
    outputs_before = metrics_before['outputs']
    outputs_after = metrics_after['outputs']
    labels = metrics_before['labels']

    # 5. Calculate per-image EPR
    epr_per_image = calculate_per_image_epr(outputs_before, outputs_after, target_idx, non_target_indices)
    
    # Calculate statistics
    epr_mean = np.mean(epr_per_image)
    epr_std = np.std(epr_per_image)
    epr_var = np.var(epr_per_image)  # 添加方差
    epr_median = np.median(epr_per_image)
    epr_min = np.min(epr_per_image)
    epr_max = np.max(epr_per_image)
    
    # 6. Compute overall metrics (for backward compatibility)
    nls, delta_target, delta_non_targets_mean = nls_metric(metrics_before['acc'].cpu(), metrics_after['acc'].cpu(), target_idx, non_target_indices)
    acc_target_before, acc_target_after = metrics_before['acc'][target_idx].cpu().item(), metrics_after['acc'][target_idx].cpu().item()
    flip_rate = (metrics_before['preds'][:, target_idx] != metrics_after['preds'][:, target_idx]).float().mean().item()

    print(f"EPR per-image statistics:")
    print(f"  Mean: {epr_mean:.4f}")
    print(f"  Variance: {epr_var:.4f}")
    print(f"  Std: {epr_std:.4f}")
    print(f"  Median: {epr_median:.4f}")
    print(f"  Min: {epr_min:.4f}")
    print(f"  Max: {epr_max:.4f}")
    print(f"NLS (overall): {nls:.4f}")
    print(f"Target score mean change: {delta_target:.4f}")
    print(f"Non-target attributes mean change: {delta_non_targets_mean:.4f}")
    print(f"Target accuracy: before {acc_target_before:.3f} | after {acc_target_after:.3f}")
    print(f"Target label flip rate: {flip_rate:.3f}")

    if args.save_path:
        # Save summary statistics
        with open(args.save_path, 'w') as f:
            f.write(f"EPR Mean: {epr_mean:.6f}\n")
            f.write(f"EPR Variance: {epr_var:.6f}\n")
            f.write(f"EPR Std: {epr_std:.6f}\n")
            f.write(f"EPR Median: {epr_median:.6f}\n")
            f.write(f"EPR Min: {epr_min:.6f}\n")
            f.write(f"EPR Max: {epr_max:.6f}\n")
            f.write(f"NLS (overall): {nls:.6f}\n")
            f.write(f"Target score mean change: {delta_target:.6f}\n")
            f.write(f"Non-target mean change: {delta_non_targets_mean:.6f}\n")
            f.write(f"Target accuracy before: {acc_target_before:.6f}\n")
            f.write(f"Target accuracy after: {acc_target_after:.6f}\n")
            f.write(f"Target label flip rate: {flip_rate:.6f}\n")
        
        # Save detailed CSV with per-image EPR
        if args.save_path.endswith('.txt'):
            csv_path = args.save_path.replace('.txt', '_detailed.csv')
        else:
            csv_path = args.save_path + '_detailed.csv'
            
        import pandas as pd
        
        # Get image filenames for CSV
        img_names = [os.path.basename(img_path) for img_path in dataset_before.images]
        
        df = pd.DataFrame({
            'image_name': img_names,
            'epr': epr_per_image,
            'target_before': torch.sigmoid(outputs_before[:, target_idx]).numpy(),  # Convert logits to probs for readability
            'target_after': torch.sigmoid(outputs_after[:, target_idx]).numpy(),   # Convert logits to probs for readability
            'target_change': torch.abs(outputs_after[:, target_idx] - outputs_before[:, target_idx]).numpy()  # Raw logits difference
        })
        
        df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")

if __name__ == '__main__':
    main()
