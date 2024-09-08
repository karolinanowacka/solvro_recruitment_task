#SOLVRO KN
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from TrajectoryModel import TrajectoryModel
from PreProcessing import *
import sys
import os
from sklearn.metrics import precision_score, recall_score

def main():    
    print("loading data...")
    X_train, y_train, X_val, y_val, X_test = load_data()

    print("removing duplicates...")
    X_train, y_train = remove_duplicates(X_train, y_train)
    X_val, y_val = remove_duplicates(X_val, y_val)
    
    print("removing outliers...")
    X_train, y_train = remove_outliers(X_train, y_train)
   
    print("changing from one-hot encoding to numerical...")
    y_train = one_hot_encoding_to_numerical(y_train)
    y_val = one_hot_encoding_to_numerical(y_val)

    print("scaling data...")
    X_train = scale_data(X_train, X_train)
    X_val = scale_data(X_train, X_val)
    X_test = scale_data(X_train, X_test)

    print("creating datasets...")
    train_dataset = TensorDataset(torch.tensor(X_train, dtype = torch.double),torch.tensor(y_train, dtype = torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype = torch.double),torch.tensor(y_val, dtype = torch.long))
    #test_dataset = TrajectoryDataset(torch.tensor(X_test, dtype = torch.double))


    print("creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 7, persistent_workers = True)
    val_loader = DataLoader(val_dataset, batch_size = 32, num_workers = 7, persistent_workers = True)
    #test_loader = DataLoader(test_dataset,batch_size = 32, num_workers = 7, persistent_workers = True )

    print("initializing model...")
    model = TrajectoryModel()
    model = model.float()
    print(model)

    print("creating optimizer...")
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-4)

    print("defining loss function and f1 score metrics...")
    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy_metric = MulticlassAccuracy(num_classes = 5)
    f1_metric = MulticlassF1Score(num_classes = 5)

    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        print("training loop...")
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        num_batches = 0

        for data, labels in train_loader:
            data, labels = data.float(), labels.long()
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, labels)
        
            total_loss += loss.item()
            total_accuracy += accuracy_metric(outputs, labels).item()
            total_f1 += f1_metric(outputs, labels).item()
            num_batches += 1

            loss.backward()
            optimizer.step()

        train_loss = total_loss / num_batches
        train_accuracy = total_accuracy / num_batches
        train_f1 = total_f1 / num_batches

        print(f"epoch {epoch+1}/{num_epochs}")
        print(f"training loss: {train_loss:.4f}")
        print(f"training accuracy: {train_accuracy:.4f}")
        print(f"training F1 Score: {train_f1:.4f}")

        model.eval()
        print("validation loop...")
        total_loss = 0.0
        total_accuracy = 0.0
        total_f1 = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.float(), labels.long()
                outputs = model(data)

                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                total_accuracy += accuracy_metric(outputs, labels).item()
                total_f1 += f1_metric(outputs, labels).item()
                
                _, preds = torch.max(outputs,1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_batches += 1

        val_loss = total_loss / num_batches
        val_accuracy = total_accuracy / num_batches
        val_f1 = total_f1 / num_batches
        precision = precision_score(all_labels, all_preds, average = 'weighted')
        recall = recall_score(all_labels, all_preds, average = 'weighted')

        print(f"validation loss: {val_loss:.4f}")
        print(f"validation accuracy: {val_accuracy:.4f}")
        print(f"validation F1 score: {val_f1:.4f}")
        print(f'validation precision: {precision:.4f}')
        print(f'validation recall: {recall:.4f}')

        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    print("Training complete.")
    sys.exit()

if __name__ == '__main__':
    main()
