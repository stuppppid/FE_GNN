import argparse
import numpy as np
from Process import *
from model import FE_GNN
from train import train_and_test_AUC

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train FE-GNN model on different datasets.")
    parser.add_argument('--dataset', type=str, default='Senolytic', choices=['NCI1', 'Bace', 'Senolytic',"MUTAG","clintox","Ames","BBBP"]
    , help='Dataset name (default: NCI1)')
    parser.add_argument('--input_dim', type=int, default=26, help='Input feature dimension (default: 26)')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden channels dimension (default: 64)')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate (default: 0.005)')
    parser.add_argument('--sigma', type=float, default=1.5, help='adjusting factor (default: 1.5)')
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Input dimension: {args.input_dim}")
    print(f"Hidden dimension: {args.hidden_dim}")
    print(f"Learning rate: {args.lr}")
    print(f"adjusting factor: {args.sigma}")

    train_loader, test_loader = load_dataset(args.dataset)
    model = FE_GNN(args.input_dim, args.hidden_dim)

    f1_scores = []
    auc_scores = []
    with open('training_results.txt', 'w') as result_file:
        result_file.write("Model Name, Learning Rate, F1 Score, AUC Score\n")
        for run in range(5):  # 随机运行五次
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            np.random.seed(run)  # 设置随机种子
            F1, AUC = train_and_test_AUC(model, train_loader, test_loader, 100, lr=args.lr,sigma=args.sigma)
            f1_scores.append(F1)
            auc_scores.append(AUC)
            result_file.write(f"{model.__class__.__name__}, {args.lr:.3f}, {F1}, {AUC}\n")
            print(f"Model: {model.__class__.__name__}, Run: {run + 1}, LR: {args.lr:.3f}, F1 Score: {F1}, AUC Score: {AUC}")

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    print(f"\nAverage F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Average AUC Score: {mean_auc:.4f} ± {std_auc:.4f}")

    with open('training_results.txt', 'a') as result_file:
        result_file.write(f"\nAverage F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\n")
        result_file.write(f"Average AUC Score: {mean_auc:.4f} ± {std_auc:.4f}\n")