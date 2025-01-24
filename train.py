from Process import *
import torch.nn as nn
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from model import FE_GNN


def fre_loss(pred_class_logits, data,sigma):
    class_counts = torch.bincount(data.y, minlength=2)
    all_counts = class_counts[0]+class_counts[1]
    if class_counts[0] > class_counts[1]:
        weight = torch.tensor([1.0, sigma*(1+(class_counts[0] / (all_counts)))], device=data.y.device)
    else:
        weight = torch.tensor([sigma*(1+(class_counts[1] / (all_counts))), 1.0], device=data.y.device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    cls_loss = loss_fn(pred_class_logits, data.y)
    return cls_loss

def Test(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data in data_loader:
            pred= model(data.x, data.edge_index, data.batch)
            pred_cls = pred.argmax(dim=-1)
            correct_predictions = (pred_cls == data.y).sum().item()
            total_correct += correct_predictions
            total_samples += data.y.size(0)
            all_preds.append(pred_cls.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
            all_probs.append(pred.cpu().numpy())

    accuracy = total_correct / total_samples
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)[:, 1]

    test_f1 = f1_score(all_targets, all_preds, average='macro')
    test_auc = roc_auc_score(all_targets, all_probs)

    return accuracy, test_f1, test_auc

def train_and_test_AUC(model, train_loader, test_loader, num_epochs,lr,sigma):

    optimizer = torch.optim.Adam(model.parameters(), lr)
    #criterion = torch.nn.CrossEntropyLoss()
    epoch_losses = []
    train_accuracies = []
    test_accuracies = []
    train_f1_scores = []
    test_f1_scores = []
    train_auc_scores = []
    test_auc_scores = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        all_train_probs = []

        # Training phase.
        for data in train_loader:
            pred = model(data.x, data.edge_index, data.batch)
            #loss = criterion(pred, data.y)
            loss = fre_loss(pred,data,sigma)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            pred_cls = pred.argmax(dim=-1)
            correct_predictions = (pred_cls == data.y).sum().item()
            total_correct += correct_predictions
            total_samples += data.y.size(0)

            all_preds.append(pred_cls.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
            all_train_probs.append(pred.detach().numpy())

        epoch_loss /= len(train_loader)
        epoch_acc = total_correct / total_samples

        # Collect Predictions and True Labels
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_train_probs = np.concatenate(all_train_probs)[:, 1]

        train_f1 = f1_score(all_targets, all_preds, average='macro')
        train_auc = roc_auc_score(all_targets, all_train_probs)

        train_accuracies.append(epoch_acc)
        epoch_losses.append(epoch_loss)
        train_f1_scores.append(train_f1)
        train_auc_scores.append(train_auc)

        print(f"Epoch: {epoch}, Loss: {epoch_loss:.3f}, Training Accuracy: {epoch_acc:.3f},"
              f" F1 Score: {train_f1:.3f}, Training AUC: {train_auc:.3f}")

        # Testing phase.
        test_acc, test_f1, test_auc = Test(model, test_loader)
        test_accuracies.append(test_acc)
        test_f1_scores.append(test_f1)
        test_auc_scores.append(test_auc)


    print('======================Accuracy================================')
    print(f"Max Training Accuracy: {np.max(train_accuracies):.3f}")
    print(f"Max Test Accuracy: {np.max(test_accuracies):.3f}")
    print(f"Mean Training Accuracy: {np.mean(train_accuracies):.3f}")
    print(f"Mean Test Accuracy: {np.mean(test_accuracies):.3f}")
    print('======================F1 Score================================')
    print(f"Max Training F1 Score: {np.max(train_f1_scores):.3f}")
    print(f"Max Test F1 Score: {np.max(test_f1_scores):.3f}")
    print(f"Mean Training F1 Score: {np.mean(train_f1_scores):.3f}")
    print(f"Mean Test F1 Score: {np.mean(test_f1_scores):.3f}")
    print('======================AUC Score================================')
    print(f"Max Training AUC Score: {np.max(train_auc_scores):.3f}")
    print(f"Max Test AUC Score: {np.max(test_auc_scores):.3f}")
    print(f"Mean Training AUC Score: {np.mean(train_auc_scores):.3f}")
    print(f"Mean Test AUC Score: {np.mean(test_auc_scores):.3f}")

    F1_score = np.max(test_f1_scores)
    AUC_score = np.max(test_auc_scores)

    F1_score = np.round(F1_score, 5)
    AUC_score = np.round(AUC_score, 5)

    return F1_score,AUC_score

if __name__ == '__main__':
    train_loader, test_loader = process_NCI1()
    model = FE_GNN(37,64)
    sigma = 1.50
    lr = 0.005

    f1_scores = []
    auc_scores = []
    with open('training_results.txt', 'w') as result_file:
        result_file.write("Model Name, Learning Rate, F1 Score, AUC Score\n")
        for run in range(5):  # Randomly run five times.
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            np.random.seed(run)  # Set random seed.
            F1, AUC = train_and_test_AUC(model, train_loader, test_loader, 100, lr=lr,sigma=sigma)
            f1_scores.append(F1)
            auc_scores.append(AUC)
            result_file.write(f"{model.__class__.__name__}, {lr:.3f}, {F1}, {AUC}\n")
            print(f"Model: {model.__class__.__name__}, Run: {run + 1}, LR: {lr:.3f}, F1 Score: {F1}, AUC Score: {AUC}")
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"\nAverage F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Average AUC Score: {mean_auc:.4f} ± {std_auc:.4f}")
    with open('training_results.txt', 'a') as result_file:
        result_file.write(f"\nAverage F1 Score: {mean_f1:.4f} ± {std_f1:.4f}\n")
        result_file.write(f"Average AUC Score: {mean_auc:.4f} ± {std_auc:.4f}\n")