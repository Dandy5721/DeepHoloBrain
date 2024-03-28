import torch
from torch import nn
import os
import numpy as np
from tqdm import tqdm
from spdnet.optimizer import StiefelMetaOptimizer
from torch.optim import SGD
from torch.utils.data import DataLoader
from dataset import OASISDataset,ADataset
from collections import OrderedDict
from spdnet.spd import SPDTransform
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified, Normalize, SPDTransform_scaled, SCaled_graph, SCaled_weighted_graph
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, roc_auc_score
import warnings
import argparse
import pandas as pd


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='**')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--layer_name', type=str, default='layers1.')
parser.add_argument('--layer_name2', type=str, default='layers1')
parser.add_argument('--out_path', type=str, default='**')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--number_cls', type=int, default=2)
parser.add_argument('--spd', type=int, default=5)
parser.add_argument('--use_conda',type=str, default='cuda:0')
args = parser.parse_args()
# model_path = '/ram/USERS/bendan/ICML2024/train_results/ICML2024_hcp-adni_spd4/window=30/models_save/8_0.8812.pt'
input_dir = '*/ADNI_BOLD_SC/AAL90_FC'
label_dir = '*/ADNI_BOLD_SC/label-2cls_new.csv'
input_SC_dir = '*/ADNI_BOLD_SC/HCP_ADNI_result_spd'
pretrained_model_path = args.model_path
pretrained_model = torch.load(pretrained_model_path)

number_SPD=0
num_workers=8
state_dict = pretrained_model['model']
# layers0_params = {name.replace('layers0.', ''): param for name, param in state_dict.items() if 'layers0' in name}
# pretrained_layers0_params = {name.replace(args.layer_name, ''): param for name, param in state_dict.items() if args.layer_name2 in name}
pretrained_layers1_params = {name.replace(args.layer_name, ''): param for name, param in state_dict.items() if args.layer_name2 in name}

# layers0_params = {name: param for name, param in state_dict.items() if 'layers0' in name}

class FinetuneModel(nn.Module):
    def __init__(self, num_classes):
        super(FinetuneModel, self).__init__()
        
        # self.layers0 = nn.Sequential(
        #     SPDTransform(116, 64, 1),
        #     SPDRectified(),
        #     SPDTransform(64, 32, 1),
        #     SPDTangentSpace(vectorize_all=False),
        #     Normalize(),
        # )
        # # new_state_dict = OrderedDict()
        # # for name, param in layers0_params.items():
        # #     new_state_dict[f'layers0.{name}'] = param
        # # print("New State Dict:", new_state_dict)
        # self.layers0.load_state_dict(pretrained_layers0_params)
        self.layers1 = nn.Sequential(

            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=False),
            Normalize(),
        )
        self.layers1.load_state_dict(pretrained_layers1_params)

        # self.layers0.load_state_dict(new_state_dict)

        # self.classifier0 = nn.Linear(32 * 33 // 2, num_classes)
        self.classifier1 = nn.Linear(64 * 65 // 2, num_classes)
    def forward(self, x):
        
        # if args.spd == 0:
        # x = self.layers0(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier0(x)
        # elif args.spd == 1:
        x = self.layers1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier1(x)

        return x

num_sample = len(
            [path for path in os.listdir(input_dir) if not path.startswith('.')]
        )
num_train = num_sample // 2
num_test = num_sample - num_train
train_dataset = ADataset(
            input_dir,
            label_dir,
            input_SC_dir,
            slice=slice(num_train),
            # window=window,
            # normalize=1e-3,
            # transpose=transpose,
            # delimiter=None,
        )
train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=num_workers
        )
test_dataset = ADataset(
    input_dir,
    label_dir,
    input_SC_dir,
    slice=slice(-num_test, None)
)
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
)

finetune_model = FinetuneModel(num_classes=2)  

optimizer = SGD(finetune_model.parameters(), lr=0.005, weight_decay=1e-5, momentum=0.9)

criterion = nn.CrossEntropyLoss()
optimizer = StiefelMetaOptimizer(optimizer)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
finetune_model.to(device)

num_epochs = args.epochs  
best_metrics = {
    'best_test_acc': 0,
    'best_test_recall': 0,
    'best_test_f1': 0,
    'best_test_precision': 0
}
for epoch in range(num_epochs):
    finetune_model.train()
    data_loader = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

    for inputs, labels, sc in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = finetune_model(inputs)
        loss = criterion(outputs, labels)
        labels_pred = outputs.argmax(-1)
        
        labels, labels_pred = labels.cpu().detach().numpy(), labels_pred.cpu().detach().numpy()
        labels, labels_pred = labels[np.isin(labels, np.unique(labels_pred))], labels_pred[np.isin(labels_pred, np.unique(labels))]

        # Use zero_division='warn' to raise warnings instead of setting to 1
        acc = accuracy_score(labels, labels_pred)
        recall = recall_score(labels, labels_pred, average='weighted', zero_division='warn')
        f1 = f1_score(labels, labels_pred, average='weighted', zero_division='warn')
        pre = precision_score(labels, labels_pred, average='weighted', zero_division='warn')
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Acc: {acc:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Precision: {pre:.4f}')

    finetune_model.eval()
    test_labels, test_labels_pred = [], []
    test_probs = []
    for test_inputs, test_labels_batch, _ in test_loader:
        test_inputs, test_labels_batch = test_inputs.to(device), test_labels_batch.to(device)
        test_outputs = finetune_model(test_inputs)
        test_probs_batch = torch.softmax(test_outputs, dim=1).cpu().detach().numpy()[:, 1]  # Assuming binary classification
        test_probs.extend(test_probs_batch)
        test_labels_batch_pred = test_outputs.argmax(-1).cpu().detach().numpy()
        test_labels.extend(test_labels_batch.cpu().detach().numpy())
        test_labels_pred.extend(test_labels_batch_pred)
   
    test_acc = accuracy_score(test_labels, test_labels_pred)
    test_recall = recall_score(test_labels, test_labels_pred, average='weighted', zero_division=1)
    test_f1 = f1_score(test_labels, test_labels_pred, average='weighted', zero_division=1)
    test_pre = precision_score(test_labels, test_labels_pred, average='weighted', zero_division=1)
    test_auc = roc_auc_score(test_labels, test_probs)
    # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}, Train Recall: {recall:.4f}, Train F1: {f1:.4f}, Train Precision: {pre:.4f}, Test Acc: {test_acc:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test Precision: {test_pre:.4f}')
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}, Train Recall: {recall:.4f}, Train F1: {f1:.4f}, Train Precision: {pre:.4f}, Test Acc: {test_acc:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test Precision: {test_pre:.4f}, Test AUC: {test_auc:.4f}')

    if test_acc > best_metrics['best_test_acc']:
        best_metrics['best_test_acc'] = test_acc
        best_metrics['best_test_recall'] = test_recall
        best_metrics['best_test_f1'] = test_f1
        best_metrics['best_test_precision'] = test_pre
        best_metrics['best_test_auc'] = test_auc
df = pd.DataFrame([best_metrics])
df.to_csv(args.out_path, index=False)