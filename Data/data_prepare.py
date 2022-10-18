from dgl import DGLGraph
import pandas as pd
from rdkit.Chem import MolFromSmiles
import numpy as np
from rdkit import Chem
from dgl.data.graph_serialize import save_graphs, load_graphs, load_labels
import torch

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom, explicit_H=False, use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other']) + one_of_k_encoding(atom.GetDegree(),
                               [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]
    return np.array(results)


def graph_from_smiles(smiles):
    g = DGLGraph()

    # Add nodes
    mol = MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    atoms_feature_all = []
    for atom_index, atom in enumerate(mol.GetAtoms()):
        atom_feature = atom_features(atom).tolist()
        atoms_feature_all.append(atom_feature)
    g.ndata["atom"] = torch.tensor(atoms_feature_all)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
    g.add_edges(src_list, dst_list)
    return g


def build_dataset(dataset_smiles, label_name, smiles_name, descriptor_seq, is_descriptor=True):
    dataset_gnn = []
    labels = dataset_smiles[label_name]
    smilesList = dataset_smiles[smiles_name]
    for i, smiles in enumerate(smilesList):
        if np.isnan(labels[i]):
            continue
        try:
            g = graph_from_smiles(smiles)
            if is_descriptor:
                molecule = [smiles, g, [labels[i]], dataset_smiles.loc[i][descriptor_seq], build_mask(labels)]
            else:
                molecule = [smiles, g, [labels[i]], build_mask(labels)]
            dataset_gnn.append(molecule)
        except:
            print('{} is transformed failed!'.format(smiles))
    return dataset_gnn


def build_mask(labels_list, mask_value=100):
    mask = []
    for i in labels_list:
        if i == mask_value:
            mask.append(0)
        else:
            mask.append(1)
    return mask


def multi_task_build_dataset(dataset_smiles, labels_list, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_list]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        try:
            g = graph_from_smiles(smiles)
            mask = build_mask(labels.loc[i], mask_value=123456)
            molecule = [smiles, g, labels.loc[i], mask, split_index.loc[i]]
            dataset_gnn.append(molecule)
            print('{}/{} molecule is transformed!'.format(i + 1, molecule_number))
        except:
            print('{} is transformed failed!'.format(smiles))
            molecule_number = molecule_number - 1
            failed_molecule.append(smiles)
    print('{}({}) is transformed failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn


def built_data_and_save_for_splited(
        origin_path='example.csv',
        save_path='example.bin',
        group_path='example_group.csv',
        task_list_selected=None,
):
    '''
        origin_path: str
            origin csv data set path, including molecule name, smiles, task
        save_path: str
            graph out put path
        group_path: str
            group out put path
        task_list_selected: list
            a list of selected task
        '''
    data_origin = pd.read_csv(origin_path)
    smiles_name = 'smiles'
    data_origin = data_origin.fillna(123456)
    labels_list = [x for x in data_origin.columns if x not in ['smiles', 'group']]
    if task_list_selected is not None:
        labels_list = task_list_selected
    data_set_gnn = multi_task_build_dataset(dataset_smiles=data_origin, labels_list=labels_list,
                                            smiles_name=smiles_name)

    smiles, graphs, labels, mask, split_index = map(list, zip(*data_set_gnn))
    graph_labels = {'labels': torch.tensor(labels),
                    'mask': torch.tensor(mask)
                    }
    split_index_pd = pd.DataFrame(columns=['smiles', 'group'])
    split_index_pd.smiles = smiles
    split_index_pd.group = split_index
    split_index_pd.to_csv(group_path, index=None, columns=None)
    print('Molecules graph is saved!')
    save_graphs(save_path, graphs, graph_labels)


def standardization_np(data, mean, std):
    return (data - mean) / (std + 1e-10)


def re_standar_np(data, mean, std):
    return data * (std + 1e-10) + mean


def split_dataset_according_index(dataset, train_index, val_index, test_index, data_type='np'):
    if data_type == 'pd':
        return pd.DataFrame(dataset[train_index]), pd.DataFrame(dataset[val_index]), pd.DataFrame(dataset[test_index])
    if data_type == 'np':
        return dataset[train_index], dataset[val_index], dataset[test_index]


def load_graph_from_csv_bin_for_splited(
        bin_path='example.bin',
        group_path='example.csv',
        select_task_index=None):
    smiles = pd.read_csv(group_path, index_col=None).smiles.values
    group = pd.read_csv(group_path, index_col=None).group.to_list()
    graphs, detailed_information = load_graphs(bin_path)
    labels = detailed_information['labels']
    mask = detailed_information['mask']
    if select_task_index is not None:
        labels = labels[:, select_task_index]
        mask = mask[:, select_task_index]
    # calculate not_use index
    notuse_mask = torch.mean(mask.float(), 1).numpy().tolist()
    not_use_index = []
    for index, notuse in enumerate(notuse_mask):
        if notuse == 0:
            not_use_index.append(index)
    train_index = []
    val_index = []
    test_index = []
    for index, group_index in enumerate(group):
        if group_index == 'training' and index not in not_use_index:
            train_index.append(index)
        if group_index == 'valid' and index not in not_use_index:
            val_index.append(index)
        if group_index == 'test' and index not in not_use_index:
            test_index.append(index)
    graph_List = []
    for g in graphs:
        graph_List.append(g)
    graphs_np = np.array(graphs)
    train_smiles, val_smiles, test_smiles = split_dataset_according_index(smiles, train_index, val_index, test_index)
    train_labels, val_labels, test_labels = split_dataset_according_index(labels.numpy(), train_index, val_index,
                                                                          test_index, data_type='pd')
    train_mask, val_mask, test_mask = split_dataset_according_index(mask.numpy(), train_index, val_index, test_index,
                                                                    data_type='pd')
    train_graph, val_graph, test_graph = split_dataset_according_index(graphs_np, train_index, val_index, test_index)

    # delete the 0_pos_label and 0_neg_label
    task_number = train_labels.values.shape[1]

    train_set = []
    val_set = []
    test_set = []
    for i in range(len(train_index)):
        molecule = [train_smiles[i], train_graph[i], train_labels.values[i], train_mask.values[i]]
        train_set.append(molecule)

    for i in range(len(val_index)):
        molecule = [val_smiles[i], val_graph[i], val_labels.values[i], val_mask.values[i]]
        val_set.append(molecule)

    for i in range(len(test_index)):
        molecule = [test_smiles[i], test_graph[i], test_labels.values[i], test_mask.values[i]]
        test_set.append(molecule)
    print(len(train_set), len(val_set), len(test_set), task_number)
    return train_set, val_set, test_set, task_number







