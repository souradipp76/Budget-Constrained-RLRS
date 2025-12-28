import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Data feature map
FEATURE_INFO_MAP = {
    "icf": ['icf1', 'icf2', 'icf3', 'icf4', 'icf5'],
    "ucf": ['ucf1', 'ufc2', 'ucf3'],
    "iv": ['iv1', 'iv2', 'iv3', 'iv4', 'iv5', 'iv6', 'iv7', 'iv8', 'iv9', 'iv10', 'iv11', 'iv12'],
    "pv": ['pv1', 'pv2', 'pv3', 'pv4', 'pv5', 'pv6', 'pv7'],
    "iv+pv": ['iv1', 'iv2', 'iv3', 'iv4', 'iv5', 'iv6', 'iv7', 'iv8', 'iv9', 'iv10', 'iv11', 'iv12', 'pv1', 'pv2', 'pv3', 'pv4', 'pv5', 'pv6', 'pv7']
}

# Utility functions
def get_pos(batch_size, seq_len):
    return torch.unsqueeze(torch.arange(seq_len).repeat(batch_size, 1), dim =-1)

def get_label(label_batch, batch_size, seq_len):
    outputs = np.zeros((batch_size, seq_len))
    for i, row in enumerate(label_batch):
        outputs[i] = np.array(json.loads(row))
    return torch.tensor(outputs, dtype=torch.float32)

def get_uid(features_batch, batch_size, seq_len):
    return torch.unsqueeze(torch.tensor([[uid] * seq_len for uid in features_batch], dtype=torch.int32), dim=-1)

def get_icf(features_batch, batch_size, seq_len):
    feature_len = len(FEATURE_INFO_MAP["icf"])
    outputs = [torch.zeros(batch_size, seq_len, dtype=torch.int32) for _ in range(feature_len)]
    for i, row in enumerate(features_batch):
        feature_data = np.array(json.loads(row.replace("null", "0")), dtype=np.int32).T
        for j in range(feature_len):
            outputs[j][i] = torch.tensor(feature_data[j])
    return torch.stack(outputs, dim = -1)

def get_ucf(features_batch, batch_size, seq_len):
    feature_len = len(FEATURE_INFO_MAP["ucf"])
    outputs = [torch.zeros(batch_size, seq_len, dtype=torch.int32) for _ in range(feature_len)]
    for i, row in enumerate(features_batch):
        feature_data = np.tile(np.array(json.loads(row.replace("null","1")), dtype=np.int32),(seq_len,1)).T
        for j in range(feature_len):
            outputs[j][i] = torch.tensor(feature_data[j])
    return torch.stack(outputs, dim = -1)

def get_iv(features_batch, batch_size, seq_len):
    feature_len = len(FEATURE_INFO_MAP['iv'])
    outputs = np.zeros((batch_size, seq_len, feature_len))
    for i, row in enumerate(features_batch):
        outputs[i] = np.array(json.loads(row))
    return torch.tensor(outputs, dtype = torch.float32)

def get_pv(features_batch, batch_size, seq_len):
    feature_len = len(FEATURE_INFO_MAP['pv'])
    outputs = np.zeros((batch_size, seq_len, feature_len))
    for i, row in enumerate(features_batch):
        outputs[i] = np.array(json.loads(row))
    return torch.tensor(outputs, dtype = torch.float32)

def get_iv_and_pv(iv_batch, pv_batch, batch_size, seq_len):
    iv = get_iv(iv_batch, batch_size, seq_len)
    pv = get_pv(pv_batch, batch_size, seq_len)
    return torch.dstack((iv, pv))

def get_features(uid_batch, ucf_batch, icf_batch, pv_batch, iv_batch, batch_size, seq_len, model_type):
    if model_type == 0:
        return [get_pos(batch_size, seq_len), get_iv(iv_batch, batch_size, seq_len)]
    elif model_type == 1:
        outputs = [get_uid(uid_batch, batch_size, seq_len)]
        outputs.append(get_ucf(ucf_batch, batch_size, seq_len))
        outputs.append(get_icf(icf_batch, batch_size, seq_len))
        outputs.append(get_iv(iv_batch, batch_size, seq_len))
        outputs.append(get_pv(pv_batch, batch_size, seq_len))
        outputs = torch.cat(outputs, dim=-1)
        return [get_pos(batch_size, seq_len), outputs]
    elif model_type == 2:
        return [get_pos(batch_size, seq_len), get_iv_and_pv(iv_batch, pv_batch, batch_size, seq_len)]
    

# Dataset and DataLoader
class CSVDataGenerator(Dataset):
    def __init__(self, filename, seq_len):
        self.data = np.loadtxt(filename, delimiter='|', dtype=str)
        self.seq_len = seq_len
        self.users = set(tuple(x) for x in [json.loads(x.replace("null", "0")) for x in self.data[:, 1]])
        self.items = set(np.array([json.loads(x.replace("null", "0")) for x in self.data[:, 2]])[:,:,0].reshape(-1).tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        uid, ucf, icf, pv, iv, label = row
        return {
            "uid": int(uid),
            "ucf": ucf,
            "icf": icf,
            "pv": pv,
            "iv": iv,
            "label": label
        }

def input_generator(filename, batch_size, seq_len, model_type, shuffle=True):
    dataset = CSVDataGenerator(filename, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    for batch in dataloader:
        batch_size = len(batch['uid'])
        yield get_features(
            batch['uid'], batch['ucf'], batch['icf'], batch['pv'], batch['iv'], batch_size, seq_len, model_type
        ), get_label(batch['label'], batch_size, seq_len)
        