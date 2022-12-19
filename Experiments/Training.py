import numpy as np
from Data.data_prepare import *
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from Experiments.model import *
from Experiments.paremeters import *
import os
import pandas as pd



# fix parameters of model
args = {}
args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
args['atom_data_field'] = 'atom'
args['classification_metric_name'] = 'roc_auc'
args['regression_metric_name'] = 'r2'
# model parameter
args['num_epochs'] = 200
args['patience'] = 50
args['mode'] = 'higher'

# task name (model name)
args['task_name'] = 'admet'
args['data_name'] = 'admet'
args['times'] = 10

# selected task, generate select task index, task class, and classification_num
args['select_task_list'] = ['Respiratory toxicity','CYP2C9', 'CYP2D6', 'Caco-2 permeability','PPB']  # change
args['select_task_index'] = []
args['classification_num'] = 0
args['regression_num'] = 0
args['all_task_list'] = ['HIA','OB','p-gp inhibitor','p-gp substrates',	'BBB',
                             'Respiratory toxicity','Hepatotoxicity', 'half-life', 'CL',
                         'Cardiotoxicity1','Cardiotoxicity10', 'Cardiotoxicity30', 'Cardiotoxicity5',
                            'CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4',
                            'Acute oral toxicity (LD50)','IGC50','ESOL','logD',	'Caco-2 permeability','PPB']  # change
# generate select task index
for index, task in enumerate(args['all_task_list']):
    if task in args['select_task_list']:
        args['select_task_index'].append(index)
# generate classification_num
for task in args['select_task_list']:
    if task in ['Respiratory toxicity','CYP2C9', 'CYP2D6']:
        args['classification_num'] = args['classification_num'] + 1
    if task in ['Caco-2 permeability','PPB']:
        args['regression_num'] = args['regression_num'] + 1

# generate classification_num
if args['classification_num'] != 0 and args['regression_num'] != 0:
    args['task_class'] = 'classification_regression'
if args['classification_num'] != 0 and args['regression_num'] == 0:
    args['task_class'] = 'classification'
if args['classification_num'] == 0 and args['regression_num'] != 0:
    args['task_class'] = 'regression'
print('Classification task:{}, Regression Task:{}'.format(args['classification_num'], args['regression_num']))
args['bin_path'] = '../data/' + args['data_name'] + '.bin'
args['group_path'] = '../data/' + args['data_name'] + '_group.csv'

result_pd = pd.DataFrame(columns=args['select_task_list']+['group'] + args['select_task_list']+['group']
                         + args['select_task_list']+['group'])
all_times_train_result = []
all_times_val_result = []
all_times_test_result = []
for time_id in range(args['times']):
    set_random_seed(2020+time_id)
    one_time_train_result = []
    one_time_val_result = []
    one_time_test_result = []
    print('{}, {}/{} time'.format(args['task_name'], time_id+1, args['times']))
    train_set, val_set, test_set, task_number = load_graph_from_csv_bin_for_splited(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        select_task_index=args['select_task_index']
    )
    print(task_number)
    print("Molecule graph generation is complete !")
    train_loader = DataLoader(dataset=train_set, batch_size=128,shuffle=True,collate_fn=collate_molgraphs)
    val_loader = DataLoader(dataset=val_set, batch_size=128,shuffle=True, collate_fn=collate_molgraphs)
    test_loader = DataLoader(dataset=test_set, batch_size=128,collate_fn=collate_molgraphs)
    pos_weight_np = pos_weight(train_set, classification_num=args['classification_num'])
    loss_criterion_c = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np.to(args['device']))
    loss_criterion_r = torch.nn.MSELoss(reduction='none')
    model = MTGL_ADMET(in_feats=40, hidden_feats=64,n_tasks=task_number,classifier_hidden_feats=64, dropout=0.2)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=10**-5)
    stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'], mode=args['mode'])
    model.to(args['device'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch_heterogeneous(args, epoch, model, train_loader, loss_criterion_c, loss_criterion_r, optimizer)

        # Validation and early stop
        validation_result = run_an_eval_epoch_heterogeneous(args, model, val_loader)
        val_score = np.mean(validation_result)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {:.4f}, best validation {:.4f}'.format(
            epoch + 1, args['num_epochs'],
            val_score,  stopper.best_score)+' validation result:', validation_result)
        if early_stop:
            break
    stopper.load_checkpoint(model)
    test_score = run_an_eval_epoch_heterogeneous(args, model, test_loader)
    train_score = run_an_eval_epoch_heterogeneous(args, model, train_loader)
    val_score = run_an_eval_epoch_heterogeneous(args, model, val_loader)
    # deal result
    result = train_score + ['training'] + val_score + ['valid'] + test_score + ['test']
    result_pd.loc[time_id] = result
    print('********************************{}, {}_times_result*******************************'.format(args['task_name'], time_id+1))
    print("training_result:", train_score)
    print("val_result:", val_score)
    print("test_result:", test_score)

result_pd.to_csv('../Result/'+args['task_name']+'_result.csv', index=None)













