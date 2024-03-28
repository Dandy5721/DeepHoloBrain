import torch
import os
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD
from tqdm import tqdm
from torchinfo import summary
import numpy as np
import pandas as pd
from torch import nn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score
from dataset import RtFCDataset, FCSCDataset, ADataset
from spdnet.optimizer import StiefelMetaOptimizer
from net import MSNet, SPDNet
from mean_shift import similarity_loss
from utils import cluster_score, plot_epochs, AverageMeter
import warnings
import argparse

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='ICML2024_ADNI_spd0')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--number_cls', type=int, default=4)
parser.add_argument('--spd', type=int, default=0)
parser.add_argument('--use_conda',type=str, default='cuda:1')

args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    num_classes = 2
    num_workers = 8
    scan = None
    seed = 0
    num_folds = 5
    resume = None
    resume_log = None
    transpose = False
    batch_size = 16
    use_cuda = True
    save_result = True
    save_test_result = True
    total_epochs = args.epochs
    log_columns = [
        'epoch',
        'train_loss',
        'train_acc',
        'test_loss',
        'test_acc',
        'val_loss',
        'val_acc',
    ]

    device = torch.device(args.use_conda if use_cuda else 'cpu')

    input_dir = '*/ADNI_BOLD_SC/AAL90_FC'
    sc_dir = '*/ADNI_BOLD_SC/ADNI_result'
    # input_dir = '/ram/USERS/bendan/ACMLab_DATA/HCP-A-SC_FC/AAL_116/FC_MOTOR_FACENAME/'

    label_dir = '*/ADNI_BOLD_SC/label-2cls_new.csv'
    output_dir_name = args.output_path
    num_sample = len(
                [path for path in os.listdir(input_dir) if not path.startswith('.')]
            )
    full_dataset = ADataset(input_dir,label_dir,sc_dir,slice=slice(num_sample))
    # kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    kf = KFold(n_splits=num_folds, shuffle=False, random_state=None)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset), start=1):

        print(f'Starting fold {fold}')

        if scan:
            train_result_path = os.path.join(
                'train_results', output_dir_name.format(scan)
            )
            test_result_path = os.path.join(
                'test_results', output_dir_name.format(scan)
            )
        else:
            train_result_path = os.path.join(
                'train_results', output_dir_name
            )
            test_result_path = os.path.join(
                'test_results', output_dir_name
            )

        models_save_path = os.path.join(train_result_path, 'models_save')

        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=num_workers)

        model = SPDNet(num_classes=num_classes)
        model.to(device)

        print(train_result_path)
        if save_test_result:
            os.makedirs(test_result_path, exist_ok=True)

        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)

        if resume:
            print('resume: ' + resume)
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['opt'])
            model.load_state_dict(checkpoint['model'])

        os.makedirs(models_save_path, exist_ok=True)

        if save_result:
            pd.DataFrame(columns=log_columns).to_csv(
                os.path.join(train_result_path, 'log.csv'), mode='a', index=False
            )
            # with open(os.path.join(train_result_path, 'net.txt'), 'w') as f:
            #     size = train_subset[0][0].shape[-1]
            #     a = torch.randn(8, size, size)
            #     a = a @ a.transpose(-2, -1) + 1e-4 * torch.eye(size)
            #     model_str = str(summary(model, input_data=a, device=device, depth=4))
            #     f.write(model_str)

        criterion2 = similarity_loss
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adadelta(model.parameters())
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        optimizer = StiefelMetaOptimizer(optimizer)

        
        best_acc = 0
        best_recall = 0
        best_f1 = 0
        for epoch in range(total_epochs):
            model.train()
            losses = AverageMeter()
            purities = AverageMeter()
            nmis = AverageMeter()
            f1s = AverageMeter()
            bar = tqdm(enumerate(train_loader), total=len(train_loader))
            for inputs, targets, sc in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                feature, outputs = model(inputs,args.spd)
                distance = criterion2(feature,targets)
                print(distance)
                loss = criterion(outputs, targets)
                labels_pred = outputs.argmax(-1)
                acc = accuracy_score(targets.cpu().detach().numpy(), labels_pred.cpu().detach().numpy())
                recall = recall_score(targets.cpu().detach().numpy(), labels_pred.cpu().detach().numpy(),average='macro')
                f1 = f1_score(targets.cpu().detach().numpy(), labels_pred.cpu().detach().numpy(),average='macro')
                losses.update(loss.item())
                purities.update(acc)
                nmis.update(recall)
                f1s.update(f1)

                bar.set_description(
                    f'Loss: {losses.avg:.4f} | acc: {purities.avg:.4f} | recall: {nmis.avg:.4f} | f1: {f1s.avg:.4f}'
                )
                loss.backward()
                optimizer.step()

        # test
        model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets, sc in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                feature, outputs = model(inputs,args.spd)
                _, predicted = torch.max(outputs.data, 1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        
        acc = accuracy_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        if acc > best_acc:
            best_acc = acc
            best_recall = recall
            best_f1 = f1
            print(f'Epoch {epoch}/{total_epochs} - Accuracy: {acc:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
        fold_results.append([best_acc, best_recall, best_f1])

    
    fold_results_np = np.array(fold_results)
    mean_results = np.mean(fold_results_np, axis=0)
    std_results = np.std(fold_results_np, axis=0)

    
    all_results = np.vstack([fold_results_np, mean_results, std_results])
    np.savetxt(os.path.join(test_result_path, 'cross_validation_results.csv'), all_results, fmt='%f', delimiter=',', header='Accuracy,Recall,F1', comments='')

    print("Best cross-validation results saved.")