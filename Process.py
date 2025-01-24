import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.loader import DataLoader
from rdkit import Chem
from torch_geometric.datasets import TUDataset
from sklearn.ensemble import RandomForestRegressor

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'DU']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(),
                                           [Chem.rdchem.HybridizationType.SP,
                                            Chem.rdchem.HybridizationType.SP2,
                                            Chem.rdchem.HybridizationType.SP3,
                                            Chem.rdchem.HybridizationType.SP3D])
    return np.array(atom_features)

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)

def process_data(df):
    train_y = np.array(df['y'])
    train_y = torch.tensor(train_y)
    datas = []
    mols = [Chem.MolFromSmiles(x) for x in df['smiles']]

    for mol, label in zip(mols, train_y):
        if mol is None:
            print("Invalid SMILES representation, unable to generate molecule, skipping this sample.")
            continue
        x = []
        for atom in mol.GetAtoms():
            x.append(get_atom_features(atom))
        x = torch.tensor(np.array(x), dtype=torch.float)
        edge_index = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edge_index.append([start, end])

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            # 如果没有边，确保边索引为空
            edge_index = torch.empty((2, 0), dtype=torch.long)

            # 检查边索引和特征矩阵的大小
        if x.size(0) == 0:
            print("The graph lacks node features, skipping this sample.")
            continue
        if edge_index.size(1) == 0:
            print("The graph has no edges; the edge index is empty.")

        label = label.unsqueeze(0) if label.dim() == 0 else label
        data = Data(x=x, edge_index=edge_index, y=label)
        datas.append(data)

    return datas

def train_test_load(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    return train_loader,test_loader

def imp(dataset):
    low_data = []
    lable = []
    for data in dataset:
        low_data.append(data.x.sum(dim=0, keepdim=True).squeeze().numpy())
        lable.append(data.y.squeeze().numpy())
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(low_data, lable)
    importances = rf.feature_importances_
    for i, importance in enumerate(importances):
        print(f"Feature {i}: {importance}")
    importances = torch.tensor(importances, dtype=torch.float32)
    importances = importances / importances.mean()
    importances = importances + 1
    return importances

def imp_data(dataset):
    print('====================================================')
    print('M-score')
    low_data = []
    lable = []
    for data in dataset:
        low_data.append(data.x.sum(dim=0, keepdim=True).squeeze().numpy())
        lable.append(data.y.squeeze().numpy())
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(low_data, lable)
    importances = rf.feature_importances_
    importances = torch.tensor(importances, dtype=torch.float32)
    importances = importances / importances.mean()
    importances = importances + 1
    print(importances)
    imp_datas = []

    for data in dataset:
        imp_data = Data(x=data.x * importances[0], edge_index=data.edge_index, y=data.y)
        imp_datas.append(imp_data)

    return imp_datas

def process_ame():
    df = pd.read_csv(r"dataset\Ames.smi", header=None, sep='\t')
    df.columns = ['smiles', 'CAS_NO', 'y']
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    data = process_data(df)
    datas = imp_data(data)
    train_loader, test_loader = train_test_load(datas)
    return train_loader, test_loader

def process_bbbp():
    df = pd.read_csv(r"dataset\BBBP.csv")
    df = df.rename(columns={'p_np': 'y'})
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    data = process_data(df)
    datas = imp_data(data)
    train_loader, test_loader = train_test_load(datas)
    return train_loader, test_loader

def process_MUTAG():
    dataset = TUDataset(root=r'dataset\TUDataset', name='MUTAG')
    data = dataset[0]  # Get the first graph object.
    print()
    print(data)
    print('=============================================================')
    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'x: {data.x}')
    print(f'y: {data.y}')
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    datas = imp_data(dataset)
    train_loader,test_loader = train_test_load(datas)
    return train_loader,test_loader

def process_Bace():
    df = pd.read_csv(r"dataset\bace.csv")
    df.rename(columns={'Class': 'y', 'mol': 'smiles'}, inplace=True)
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    print(df.head())
    data = process_data(df)
    datas = imp_data(data)
    train_loader, test_loader = train_test_load(datas)
    return train_loader, test_loader

def process_clintox():
    df = pd.read_csv(r"dataset\clintox.csv")
    df.rename(columns={'CT_TOX': 'y'}, inplace=True)
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    print(df.head())
    data = process_data(df)
    datas = imp_data(data)
    train_loader, test_loader = train_test_load(datas)
    return train_loader, test_loader

def process_senolytic():
    df = pd.read_csv(r"dataset\senolytic.csv")
    df.rename(columns={'SMILES': 'smiles'}, inplace=True)
    df.rename(columns={'senolytic': 'y'}, inplace=True)
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    print(df.head())
    data = process_data(df)
    datas = imp_data(data)
    train_loader, test_loader = train_test_load(datas)
    return train_loader, test_loader

def process_NCI1():
    dataset = TUDataset(root=r'dataset\TUDataset', name='NCI1')
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    data = dataset[0]  # Get the first graph object.
    print()
    print(data)
    print('=============================================================')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')
    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'x: {data.x}')
    print(f'y: {data.y}')
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    train_loader,test_loader = train_test_load(dataset)
    print('=============================================================')
    return train_loader,test_loader

def load_dataset(dataset_name):
    if dataset_name == "NCI1":
        return process_NCI1()
    elif dataset_name == "Senolytic":
        return process_senolytic()
    elif dataset_name == "Bace":
        return process_Bace()
    elif dataset_name == "MUTAG":
        return process_MUTAG()
    elif dataset_name == "clintox":
        return process_clintox()
    elif dataset_name == "Ames":
        return process_ame()
    elif dataset_name == "BBBP":
        return process_bbbp()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
if __name__ == '__main__':
    df = process_Bace()
