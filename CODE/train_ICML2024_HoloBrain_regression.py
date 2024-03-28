import torch
import os
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
from torchinfo import summary
import numpy as np
import pandas as pd
from torch import nn
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_absolute_error, r2_score,mean_squared_error
from dataset import RtFCDataset, FCSCDataset, FC_SCDataset, FC_SCDataset2, HCPADataset_pre
from spdnet.optimizer import StiefelMetaOptimizer
from net import MSNet, SPDNet, DeepHoloBrain
from mean_shift import similarity_loss
from utils import cluster_score, plot_epochs, AverageMeter
import warnings
import argparse

warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--fc_path', default='*')
parser.add_argument('--sc_path', default='*')
parser.add_argument('--label_dir', default='*/mocal_score.csv')
parser.add_argument('--output_path', type=str, default='ICML2024_DeepHoloBrain_regression1')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--number_cls', type=int, default=1)
parser.add_argument('--spd', type=int, default=1)
parser.add_argument('--use_conda',type=str, default='cuda:1')


args = parser.parse_args()

if __name__ == '__main__':
    num_classes = args.number_cls
    windows = [30]
    num_workers = 8
    scan = None
    # seed = 10

    resume = None
    resume_log = None
    transpose = False

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

    input_dir = args.fc_path
    sc_dir = args.sc_path
    label_dir = args.label_dir
    output_dir_name = args.output_path

    for window in windows:
        # torch.manual_seed(seed)
        # np.random.seed(seed)

        print('\n############### current window = %d ###############\n' % window)

        if scan:
            train_result_path = os.path.join(
                'train_results', output_dir_name, '{}/window={}'.format(scan, window)
            )
            test_result_path = os.path.join(
                'test_results', output_dir_name, '{}/window={}'.format(scan, window)
            )
        else:
            train_result_path = os.path.join(
                'train_results', output_dir_name, 'window={}'.format(window)
            )
            test_result_path = os.path.join(
                'test_results', output_dir_name, 'window={}'.format(window)
            )

        models_save_path = os.path.join(train_result_path, 'models_save')

        num_sample = len(
            [path for path in os.listdir(input_dir) if not path.startswith('.')]
        )
        num_train = num_sample // 10 * 7
        num_val = (num_sample - num_train) // 2
        num_test = num_sample - num_train - num_val
        # num_train = 2
        # num_val = 2
        # num_test = 2
        print(f'num_sample = {num_sample}')
        print(f'num_train = {num_train}')
        print(f'num_val = {num_val}')
        print(f'num_test = {num_test}')

        train_dataset = HCPADataset_pre(
            input_dir,
            label_dir,
            sc_dir,
            slice=slice(num_train),
            # window=window,
            # normalize=1e-3,
            # transpose=transpose,
            # delimiter=None,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=num_workers
        )

        val_dataset = HCPADataset_pre(
            input_dir,
            label_dir,
            sc_dir,
            slice=slice(num_train, num_train + num_val),
            # window=window,
            # normalize=1e-3,
            # transpose=transpose,
            # delimiter=None,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
        )

        test_dataset = HCPADataset_pre(
            input_dir,
            label_dir,
            sc_dir,
            slice=slice(-num_test, None),
            # window=window,
            # normalize=1e-3,
            # transpose=transpose,
            # delimiter=None,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
        )

        model = DeepHoloBrain(num_classes=num_classes)
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
            #     size = train_dataset[0][0].shape[-1]
            #     a = torch.randn(8, size, size)
            #     a = a @ a.transpose(-2, -1) + 1e-4 * torch.eye(size)
            #     model_str = str(summary(model, input_data=a,sc=a, device=device, depth=4))
            #     f.write(model_str)

        # criterion = similarity_loss
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        # optimizer = torch.optim.Adadelta(model.parameters())
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        optimizer = StiefelMetaOptimizer(optimizer)

        def train(data_loader):
            model.train()
            losses = AverageMeter()
            purities = AverageMeter()
            nmis = AverageMeter()
            f1s = AverageMeter()
            bar = tqdm(enumerate(data_loader), total=len(data_loader))
            for batch_idx, (inputs, targets, sc) in bar:
                inputs = inputs.squeeze().to(device)
                # targets = targets.squeeze().to(device)
                targets = targets.to(device)
                scs = sc.squeeze().to(device)
                optimizer.zero_grad()
                # print(inputs.shape)
                # print(targets.shape)
                # print(scs.shape)
                feature, outputs, row, column = model(inputs, scs, args.spd)

                loss1 = criterion(outputs, targets)
                loss = loss1 + torch.abs(feature)

                loss.backward()
                optimizer.step()
                
                # clustering = SpectralClustering(
                #     n_clusters=num_classes, n_jobs=-1, gamma=1
                # )
                # labels_pred = clustering.fit(outputs.cpu().detach().numpy()).labels_
                labels_pred = outputs.argmax(-1)
                acc = mean_absolute_error(targets.cpu().detach().numpy(),outputs.cpu().detach().numpy())
                recall =r2_score(targets.cpu().detach().numpy(),outputs.cpu().detach().numpy())
                f1 = mean_squared_error(targets.cpu().detach().numpy(),outputs.cpu().detach().numpy())
                # print("Targets length:", len(targets))
                # print("Predictions length:", len(labels_pred))
                # acc = accuracy_score(targets.cpu().detach().numpy(), labels_pred.cpu().detach().numpy())
                # # purity, _, nmi = cluster_score(targets.cpu(), labels_pred)
                # recall = recall_score(targets.cpu().detach().numpy(), labels_pred.cpu().detach().numpy(),average='macro')
                # f1 = f1_score(targets.cpu().detach().numpy(), labels_pred.cpu().detach().numpy(),average='macro')
                losses.update(loss.item())
                purities.update(acc)
                nmis.update(recall)
                f1s.update(f1)

                bar.set_description(
                    f'Loss: {losses.avg:.4f} | acc: {purities.avg:.4f} | recall: {nmis.avg:.4f} | f1: {f1s.avg:.4f}'
                )
            return losses.avg, purities.avg

        @torch.no_grad()
        def test(data_loader):
            model.eval()
            losses = AverageMeter()
            purities = AverageMeter()
            nmis = AverageMeter()
            f1s = AverageMeter()
            bar = tqdm(enumerate(data_loader), total=len(data_loader))

            all_targets = []
            all_predictions = []

            for batch_idx, (inputs, targets, sc) in bar:
                inputs = inputs.squeeze().to(device)
                # targets = targets.squeeze().to(device)
                targets = targets.to(device)
                scs = sc.squeeze().to(device)
                feature, outputs, row, column = model(inputs, scs, args.spd)
                loss1 = criterion(outputs, targets)
                loss = loss1 + torch.abs(feature)
                labels_pred = outputs.argmax(-1)

                all_targets.extend(targets.cpu().detach().numpy().flatten())

                all_predictions.extend(labels_pred.cpu().detach().numpy().flatten())

                losses.update(loss.item())
                bar.set_description(f'Loss: {losses.avg:.4f}')

            acc = mean_absolute_error(all_targets, all_predictions)
            recall = r2_score(all_targets, all_predictions)
            f1 = mean_squared_error(all_targets, all_predictions)
            print(acc)
            purities.update(acc)
            nmis.update(recall)
            f1s.update(f1)
            bar.set_description(
                                f'Loss: {losses.avg:.4f} | acc: {acc:.4f} | recall: {recall:.4f} | f1: {f1:.4f}'
                            )
            return losses.avg, purities.avg, all_predictions, acc, recall, f1, row, column


            # return losses.avg, purities.avg, labels, all_acc, all_recall, all_f1

        train_losses = []
        train_accs = []

        test_losses = []
        test_accs = []

        val_losses = []
        val_accs = []

        epochs = []
        best_acc = 0

        start_epoch = 1

        if resume_log:
            print('resume_log: ' + resume_log)
            log = pd.read_csv(resume_log)
            train_losses, train_accs = (
                log['train_loss'].to_list(),
                log['train_acc'].to_list(),
            )
            test_losses, test_accs = (
                log['test_loss'].to_list(),
                log['test_acc'].to_list(),
            )
            val_losses, val_accs = log['val_loss'].to_list(), log['val_acc'].to_list()
            epochs = log['epoch'].to_list()
            best_acc = max(test_accs)
            start_epoch = max(epochs) + 1

            for i, epoch in enumerate(epochs):
                print('\nEpoch: %d' % epoch)
                print('Loss: %.4f | acc: %.4f' % (train_losses[i], train_accs[i]))
                print('Loss: %.4f | acc: %.4f' % (test_losses[i], test_accs[i]))
                print('Loss: %.4f | acc: %.4f' % (val_losses[i], val_accs[i]))

        for epoch in range(start_epoch, start_epoch + total_epochs):
            print('\nEpoch: %d' % epoch)
            train_loss, train_acc = train(train_loader)
            test_loss, test_acc, labels, accs, recalls, f1s, row, column = test(test_loader)
            val_loss, val_acc, _, _, _, _, _, _ = test(val_loader)
            results = np.array([[accs, recalls, f1s]])

            if val_acc > best_acc:
                print('best')
                best_acc = test_acc
                if save_test_result:
                    # for i in range(len(labels)):
                    #     np.savetxt(
                    #         os.path.join(test_result_path, '%d.csv') % (i + 1),
                    #         labels[i],
                    #         fmt='%d',
                    #         delimiter=',',
                    #     )
                    np.savetxt(
                        os.path.join(test_result_path, 'accs.csv'),
                        results,
                        fmt='%f',
                        delimiter=',',
                        header='Accuracy,Recall,F1-Score',
                        comments=''
                    )
                    np.savetxt(
                        os.path.join(test_result_path, 'attention_row.csv'),
                        row.cpu().numpy(),
                        fmt='%f',
                        delimiter=',',
                    )
                    np.savetxt(
                        os.path.join(test_result_path, 'attention_column.csv'),
                        column.cpu().numpy(),
                        fmt='%f',
                        delimiter=',',
                    )


            epochs.append(epoch)

            train_losses.append(train_loss)
            train_accs.append(train_acc)

            test_losses.append(test_loss)
            test_accs.append(test_acc)

            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if save_result:
                plot_epochs(
                    os.path.join(train_result_path, 'loss.svg'),
                    [train_losses, test_losses, val_losses],
                    epochs,
                    xlabel='epoch',
                    ylabel='loss',
                    legends=['train', 'test', 'val'],
                    max=False,
                )
                plot_epochs(
                    os.path.join(train_result_path, 'acc.svg'),
                    [train_accs, test_accs, val_accs],
                    epochs,
                    xlabel='epoch',
                    ylabel='accuracy',
                    legends=['train', 'test', 'val'],
                )

                pd.DataFrame(
                    [
                        [
                            epoch,
                            train_loss,
                            train_acc,
                            test_loss,
                            test_acc,
                            val_loss,
                            val_acc,
                        ]
                    ]
                ).to_csv(
                    os.path.join(train_result_path, 'log.csv'),
                    mode='a',
                    index=False,
                    header=False,
                )
                torch.save(
                    {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'opt': optimizer.state_dict(),
                    },
                    os.path.join(models_save_path, "%d_%.4f.pt" % (epoch, test_acc)),
                )
