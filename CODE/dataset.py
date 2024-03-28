import os
import numpy as np
import torch
import scipy.io
import pandas as pd
from torch.utils.data import Dataset
from utils import (
    sorted_aphanumeric,
    fc2vector,
    sliding_window_corrcoef,
)


class RtFCDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        label_dir=None,
        window=None,
        slice=None,
        normalize=False,
        vectorize=False,
        transpose=True,
        delimiter=',',
    ):
        super(RtFCDataset, self).__init__()
        self.normalize = normalize
        self.vectorize = vectorize
        self.transpose = transpose
        self.delimiter = delimiter
        self.data_path = []
        self.labels = []

        if data_dir is not None:
            self.data_path = [
                os.path.join(data_dir, name)
                for name in sorted_aphanumeric(os.listdir(data_dir))
            ]
            if slice:
                self.data_path = self.data_path[slice]

        self.n_data = len(self.data_path)
        self.window = [window] * self.n_data

        if isinstance(label_dir, str):
            self.labels = [
                torch.from_numpy(np.loadtxt(label_dir, dtype=np.int64))
            ] * self.n_data
        elif label_dir is not None:
            self.labels = label_dir

        if self.n_data > 0:
            print('number of samples:', self.n_data)
            print('window:', window)
            print('data_path:', f'{self.data_path[0]},...,{self.data_path[-1]}')
            if isinstance(label_dir, str):
                print('label_dir:', label_dir)

    def __len__(self):
        return self.n_data

    def __add__(self, other):
        self.data_path = list(self.data_path) + list(other.data_path)
        self.labels = list(self.labels) + list(other.labels)
        self.window = list(self.window) + list(other.window)
        self.n_data += other.n_data
        return self

    def __getitem__(self, idx):
        path = self.data_path[idx]
        load = np.loadtxt
        # print(load)
        loaded_data = load(path, delimiter=',', dtype=np.float32, skiprows=1)
        data = loaded_data[1:, 1:]  # Slice from row index 1 onwards and column index 1 onwards
        # if self.transpose:
        data = data.T
        data = sliding_window_corrcoef(data, self.window[idx])

        # data = sliding_window_corrcoef(
        #     load(path, delimiter=self.delimiter, dtype=np.float32).T
        #     if self.transpose
        #     else load(path, delimiter=self.delimiter, dtype=np.float32),
        #     self.window[idx],
        # )
        data = torch.from_numpy(data)

        if self.normalize:
            data += self.normalize * torch.eye(data.shape[-1])
        if self.vectorize:
            data = fc2vector(data)
        num_rows = data.size(0)
        label = torch.zeros(num_rows, device=data.device, dtype=data.dtype)
        print(label.shape)
        # label = self.labels[idx]

        return data, label

class FCSCDataset(Dataset):
    def __init__(self, data_dir, delimiter=',',slice=None):
        super(FCSCDataset, self).__init__()
        self.data_dir = data_dir
        self.delimiter = delimiter
        self.data_path = []
        self.sc_path = []
        self.labels = []

        # Populate data_path and labels
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]
            self.labels = self.labels[slice]

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_dir, filename)
                self.data_path.append(file_path)
                
                if 'CARIT' in filename:
                    self.labels.append(np.int64(2))
                elif 'FACENAME' in filename:
                    self.labels.append(np.int64(0))
                elif 'VISMOTOR' in filename:
                    self.labels.append(np.int64(1))
                elif 'REST' in filename:
                    self.labels.append(np.int64(3))
                else:
                    self.labels.append(np.int64(4))  # Default label for unidentified cases

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        data = pd.read_csv(self.data_path[idx], delimiter=self.delimiter, header=0, usecols=lambda column: column != 'Unnamed: 0').fillna(0).values
        data = torch.from_numpy(data).float()

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        return data, label
    
class FC_SCDataset(Dataset):
    def __init__(self, data_dir, sc_dir, delimiter=',',slice=None):
        super(FC_SCDataset, self).__init__()
        self.data_dir = data_dir
        self.sc_dir = sc_dir
        self.delimiter = delimiter
        self.data_path = []
        self.sc_path = []
        self.labels = []

        # Populate data_path and labels
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]
            self.sc_path = self.sc_path[slice]
            self.labels = self.labels[slice]

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_dir, filename)
                self.data_path.append(file_path)

                if 'CARIT' in filename:
                    self.labels.append(np.int64(1))
                elif 'FACENAME' in filename:
                    self.labels.append(np.int64(2))
                elif 'VISMOTOR' in filename:
                    self.labels.append(np.int64(3))
                elif 'REST' in filename:
                    self.labels.append(np.int64(0))
                else:
                    self.labels.append(np.int64(4))  # Default label for unidentified cases

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        data = pd.read_csv(self.data_path[idx], delimiter=self.delimiter, header=0, usecols=lambda column: column != 'Unnamed: 0').fillna(0).values
        data = torch.from_numpy(data).float()
        # print(self.data_path[idx])
        # print(self.sc_path)
        # sc_data = scipy.io.loadmat(self.sc_path[idx])
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        sc_folder_name = os.path.basename(self.data_path[idx])[:14]
        sc_folder_path = os.path.join(self.sc_dir, sc_folder_name)
        # print(f'sc_path = {sc_folder_path}')
        sc_data = None
        if os.path.isdir(sc_folder_path):
            for sc_filename in os.listdir(sc_folder_path):
                if sc_filename.endswith('.mat'):
                    sc_file_path = os.path.join(sc_folder_path, sc_filename)
                    # print(f'sc_path = {sc_file_path}')
                    sc_data = scipy.io.loadmat(sc_file_path)
                    sc_data = sc_data['aal116_sift_radius2_count_connectivity']
                    row_sums = sc_data.sum(axis=1)
                    row_sums[row_sums == 0] = 1  
                    sc_data = sc_data / row_sums[:, np.newaxis]
                    # print(sc_data)
                    sc_data = torch.from_numpy(sc_data).float()

                    break
        if sc_data is None:
            print(f"Warning: SC data not found for {self.data_path[idx]}")
            sc_data = scipy.io.loadmat('**')
            sc_data = sc_data['aal116_sift_radius2_count_connectivity']
            row_sums = sc_data.sum(axis=1)
            row_sums[row_sums == 0] = 1  
            sc_data = sc_data / row_sums[:, np.newaxis]
            sc_data = torch.from_numpy(sc_data).float()
        # else:
        #     # print(f"Warning: SC data not found for {self.data_path[idx]}")
        #     return None
        # mask = (label == 2) | (label == 3)
        # data = data[mask]
        # label = label[mask]

        # print(f'sc = {sc_data}')
        # if label.item() in [2, 3]:
        #     return data, label, sc_data
        # else:
        #     # If label is not 2 or 3, return None (item will be skipped)
        #     return None
        return data, label, sc_data
    
class FC_SCDataset2(Dataset):
    def __init__(self, data_dir, sc_dir, delimiter=',',slice=None):
        super(FC_SCDataset2, self).__init__()
        self.data_dir = data_dir
        self.sc_dir = sc_dir
        self.delimiter = delimiter
        self.data_path = []
        self.sc_path = []
        self.labels = []

        # Populate data_path and labels
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]
            self.sc_path = self.sc_path[slice]
            self.labels = self.labels[slice]

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_dir, filename)
                self.data_path.append(file_path)

                if 'FACENAME' in filename:
                    self.labels.append(np.int64(0))
                elif 'VISMOTOR' in filename:
                    self.labels.append(np.int64(1))
                else:
                    self.labels.append(np.int64(2))  # Default label for unidentified cases

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        data = pd.read_csv(self.data_path[idx], delimiter=self.delimiter, header=0, usecols=lambda column: column != 'Unnamed: 0').fillna(0).values
        data = torch.from_numpy(data).float()
        # print(self.data_path[idx])
        # print(self.sc_path)
        # sc_data = scipy.io.loadmat(self.sc_path[idx])
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        sc_folder_name = os.path.basename(self.data_path[idx])[:14]
        sc_folder_path = os.path.join(self.sc_dir, sc_folder_name)
        # print(f'sc_path = {sc_folder_path}')
        sc_data = None
        if os.path.isdir(sc_folder_path):
            for sc_filename in os.listdir(sc_folder_path):
                if sc_filename.endswith('.mat'):
                    sc_file_path = os.path.join(sc_folder_path, sc_filename)
                    # print(f'sc_path = {sc_file_path}')
                    sc_data = scipy.io.loadmat(sc_file_path)
                    sc_data = sc_data['aal116_sift_radius2_count_connectivity']
                    row_sums = sc_data.sum(axis=1)
                    row_sums[row_sums == 0] = 1  
                    sc_data = sc_data / row_sums[:, np.newaxis]
                    # print(sc_data)
                    sc_data = torch.from_numpy(sc_data).float()

                    break
        if sc_data is None:
            print(f"Warning: SC data not found for {self.data_path[idx]}")
            sc_data = scipy.io.loadmat('**')
            sc_data = sc_data['aal116_sift_radius2_count_connectivity']
            row_sums = sc_data.sum(axis=1)
            row_sums[row_sums == 0] = 1  
            sc_data = sc_data / row_sums[:, np.newaxis]
            sc_data = torch.from_numpy(sc_data).float()
        # else:
        #     # print(f"Warning: SC data not found for {self.data_path[idx]}")
        #     return None
        # mask = (label == 2) | (label == 3)
        # data = data[mask]
        # label = label[mask]

        # print(f'sc = {sc_data}')
        # if label.item() in [2, 3]:
        #     return data, label, sc_data
        # else:
        #     # If label is not 2 or 3, return None (item will be skipped)
        #     return None
        # print(label)
        return data, label, sc_data
class FC_SCDataset2(Dataset):
    def __init__(self, data_dir, sc_dir, delimiter=',',slice=None):
        super(FC_SCDataset2, self).__init__()
        self.data_dir = data_dir
        self.sc_dir = sc_dir
        self.delimiter = delimiter
        self.data_path = []
        self.sc_path = []
        self.labels = []

        # Populate data_path and labels
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]
            self.sc_path = self.sc_path[slice]
            self.labels = self.labels[slice]

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_dir, filename)
                self.data_path.append(file_path)

                if 'FACENAME' in filename:
                    self.labels.append(np.int64(0))
                elif 'VISMOTOR' in filename:
                    self.labels.append(np.int64(1))
                else:
                    self.labels.append(np.int64(2))  # Default label for unidentified cases

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        data = pd.read_csv(self.data_path[idx], delimiter=self.delimiter, header=0, usecols=lambda column: column != 'Unnamed: 0').fillna(0).values
        data = torch.from_numpy(data).float()
        # print(self.data_path[idx])
        # print(self.sc_path)
        # sc_data = scipy.io.loadmat(self.sc_path[idx])
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)
        sc_folder_name = os.path.basename(self.data_path[idx])[:14]
        sc_folder_path = os.path.join(self.sc_dir, sc_folder_name)
        # print(f'sc_path = {sc_folder_path}')
        sc_data = None
        if os.path.isdir(sc_folder_path):
            for sc_filename in os.listdir(sc_folder_path):
                if sc_filename.endswith('.mat'):
                    sc_file_path = os.path.join(sc_folder_path, sc_filename)
                    # print(f'sc_path = {sc_file_path}')
                    sc_data = scipy.io.loadmat(sc_file_path)
                    sc_data = sc_data['aal116_sift_radius2_count_connectivity']
                    row_sums = sc_data.sum(axis=1)
                    row_sums[row_sums == 0] = 1  
                    sc_data = sc_data / row_sums[:, np.newaxis]
                    # print(sc_data)
                    sc_data = torch.from_numpy(sc_data).float()

                    break
        if sc_data is None:
            print(f"Warning: SC data not found for {self.data_path[idx]}")
            sc_data = scipy.io.loadmat('**')
            sc_data = sc_data['aal116_sift_radius2_count_connectivity']
            row_sums = sc_data.sum(axis=1)
            row_sums[row_sums == 0] = 1  
            sc_data = sc_data / row_sums[:, np.newaxis]
            sc_data = torch.from_numpy(sc_data).float()
        # else:
        #     # print(f"Warning: SC data not found for {self.data_path[idx]}")
        #     return None
        # mask = (label == 2) | (label == 3)
        # data = data[mask]
        # label = label[mask]

        # print(f'sc = {sc_data}')
        # if label.item() in [2, 3]:
        #     return data, label, sc_data
        # else:
        #     # If label is not 2 or 3, return None (item will be skipped)
        #     return None
        # print(label)
        return data, label, sc_data
            
class ADataset(Dataset):
    def __init__(self, data_dir, label_path, sc_dir, delimiter=',', slice=None):
        super(ADataset, self).__init__()
        self.data_dir = data_dir
        self.label_path = label_path
        self.sc_dir = sc_dir
        self.delimiter = delimiter
        self.data_path = []
        self.sc_path = []

        # Populate data_path and sc_path
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]
            self.sc_path = self.sc_path[slice]

        self.labels = pd.read_csv(self.label_path, header=0)
        # print(self.labels)

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.data_dir, filename)
                self.data_path.append(file_path)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data = pd.read_csv(self.data_path[idx], delimiter=' ', header=None).fillna(0).values
        # print(data.shape)
        data = torch.from_numpy(data).float()

        # label = self.labels.iloc[idx, 1]
        file_name = os.path.basename(self.data_path[idx])[4:12]
        # print(file_name)
        matching_row = self.labels[self.labels['subject_id'].str[:8] == file_name]
        # print(matching_row)
        if not matching_row.empty:
            label = matching_row.iloc[0]['Label1']
            # print("Label:", label)
        else:
            # print("No matching label found.")
            label = 0
        
        label = torch.tensor(label, dtype=torch.long)

        sc_folder_name = os.path.basename(self.data_path[idx])[:12]
        # print(sc_folder_name)
        sc_data = None
        if os.path.isdir(self.sc_dir):
            for sc_filename in os.listdir(self.sc_dir):
                if sc_filename.endswith('.mat') and sc_filename[:12] == sc_folder_name:
                    sc_file_path = os.path.join(self.sc_dir, sc_filename)
                    # print(sc_file_path)
                    sc_data = scipy.io.loadmat(sc_file_path)
                    sc_data = sc_data['aal116_sift_radius2_count_connectivity']
                    row_sums = sc_data.sum(axis=1)
                    row_sums[row_sums == 0] = 1  
                    sc_data = sc_data / row_sums[:, np.newaxis]
                    sc_data = torch.from_numpy(sc_data).float()
                    # print(sc_data.shape)
                    break  

        if sc_data is None:
            # print(f"Warning: SC data not found for {self.data_path[idx]}")
            sc_data = scipy.io.loadmat('**')
            sc_data = sc_data['aal116_sift_radius2_count_connectivity']
            row_sums = sc_data.sum(axis=1)
            row_sums[row_sums == 0] = 1  
            sc_data = sc_data / row_sums[:, np.newaxis]
            sc_data = torch.from_numpy(sc_data).float()
            # print(sc_data.shape)

        return data, label, sc_data

class OASISDataset(Dataset):
    def __init__(self, data_dir, label_path, sc_dir, delimiter=',', slice=None):
        super(OASISDataset, self).__init__()
        self.data_dir = data_dir
        self.label_path = label_path
        self.sc_dir = sc_dir
        self.delimiter = delimiter
        self.data_path = []
        self.sc_path = []

        # Populate data_path and sc_path
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]
            self.sc_path = self.sc_path[slice]

        self.labels = pd.read_csv(self.label_path, header=0)
        # print(self.labels)

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.data_dir, filename)
                self.data_path.append(file_path)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data = pd.read_csv(self.data_path[idx], delimiter=' ', header=None).fillna(0).values
        # print(data.shape)
        data = torch.from_numpy(data).float()
        
        file_name = os.path.basename(self.data_path[idx])[:8]
        # print(file_name)
        matching_row = self.labels[self.labels['SUBJECT_ID'].str[:8] == file_name]
        # print(matching_row)
        if not matching_row.empty:
            label = matching_row.iloc[0]['Label1']
            # print("Label:", label)
        else:
            # print("No matching label found.")
            label = 0
     
        # label = self.labels.iloc[idx, 1]

        label = torch.tensor(label, dtype=torch.long)

        sc_folder_name = os.path.basename(self.data_path[idx])[:8]
        # print(sc_folder_name)
        sc_data = None
        if os.path.isdir(self.sc_dir):
            for sc_filename in os.listdir(self.sc_dir):
                if sc_filename.endswith('.csv') and sc_filename[:8] == sc_folder_name:
                    sc_file_path = os.path.join(self.sc_dir, sc_filename)
                    # print(sc_file_path)
                    # sc_data = scipy.io.loadmat(sc_file_path)
                    sc_data = pd.read_csv(sc_file_path,header=None)
                    # print(sc_data)
                    # sc_data = sc_data['aal116_sift_radius2_count_connectivity']
                    row_sums = sc_data.sum(axis=1)
                    row_sums[row_sums == 0] = 1

                    
                    sc_data_np = sc_data.to_numpy()
                    row_sums_np = row_sums.to_numpy()

                    
                    sc_data = torch.from_numpy(sc_data_np / row_sums_np[:, np.newaxis]).float()
                    break  

        if sc_data is None:
            # print(f"Warning: SC data not found for {self.data_path[idx]}")
            sc_data = pd.read_csv('**',header=None)
            # sc_data = sc_data['aal116_sift_radius2_count_connectivity']
            # print(sc_data)
            row_sums = sc_data.sum(axis=1)
            row_sums[row_sums == 0] = 1

            
            sc_data_np = sc_data.to_numpy()
            row_sums_np = row_sums.to_numpy()

            
            sc_data = torch.from_numpy(sc_data_np / row_sums_np[:, np.newaxis]).float()


        return data, label, sc_data
    
class HCPADataset_pre(Dataset):
    def __init__(self, data_dir, label_path, sc_dir, delimiter=',', slice=None):
        super(HCPADataset_pre, self).__init__()
        self.data_dir = data_dir
        self.label_path = label_path
        self.sc_dir = sc_dir
        self.delimiter = delimiter
        self.data_path = []
        self.sc_path = []

        # Populate data_path and sc_path
        self._load_data_paths()

        if slice is not None:
            self.data_path = self.data_path[slice]
            self.sc_path = self.sc_path[slice]

        self.labels = pd.read_csv(self.label_path, header=0)
        # print(self.labels)

    def _load_data_paths(self):
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.data_dir, filename)
                self.data_path.append(file_path)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data = pd.read_csv(self.data_path[idx], delimiter=self.delimiter, header=0, usecols=lambda column: column != 'Unnamed: 0').fillna(0).values
        # data = torch.from_numpy(data).float()
        data = torch.from_numpy(data).float()
        
        file_name = os.path.basename(self.data_path[idx])[4:14]
        # print(file_name)
        # print(self.labels['src_subject_id'].str[:10])
        matching_row = self.labels[self.labels['src_subject_id'].str[:10] == file_name]
        # print(matching_row)
        if not matching_row.empty:
            label = matching_row.iloc[0]['moca_total']
            # print("Label:", label)
        else:
            # print("No matching label found.")
            # print(matching_row)
            label = 28
     
        # label = self.labels.iloc[idx, 1]

        label = torch.tensor(label).float()

        sc_folder_name = os.path.basename(self.data_path[idx])[:14]
        sc_folder_path = os.path.join(self.sc_dir, sc_folder_name)
        # print(sc_folder_name)
        sc_data = None
        if os.path.isdir(sc_folder_path):
            for sc_filename in os.listdir(sc_folder_path):
                if sc_filename.endswith('.mat'):
                    sc_file_path = os.path.join(sc_folder_path, sc_filename)
                    # print(f'sc_path = {sc_file_path}')
                    sc_data = scipy.io.loadmat(sc_file_path)
                    sc_data = sc_data['aal116_sift_radius2_count_connectivity']
                    row_sums = sc_data.sum(axis=1)
                    row_sums[row_sums == 0] = 1  
                    sc_data = sc_data / row_sums[:, np.newaxis]
                    # print(sc_data)
                    sc_data = torch.from_numpy(sc_data).float()

                    break
        if sc_data is None:
            print(f"Warning: SC data not found for {self.data_path[idx]}")
            sc_data = scipy.io.loadmat('**')
            sc_data = sc_data['aal116_sift_radius2_count_connectivity']
            row_sums = sc_data.sum(axis=1)
            row_sums[row_sums == 0] = 1  
            sc_data = sc_data / row_sums[:, np.newaxis]
            sc_data = torch.from_numpy(sc_data).float()


        return data, label, sc_data