import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def run_test(model, test_loader, device, model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='micro', zero_division=0)

    return accuracy, precision, recall, f1
