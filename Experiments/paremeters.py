from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score
import torch
import torch.nn.functional as F
import dgl
import numpy as np
import random
import pandas as pd


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def collate_molgraphs(data):
    smiles, graphs, labels, mask = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.tensor(labels)
    mask = torch.tensor(mask)

    return smiles, bg, labels,  mask

def pos_weight(train_set, classification_num):
    smiles, graphs, labels, mask = map(list, zip(*train_set))
    labels = np.array(labels)
    task_pos_weight_list = []
    for task in range(classification_num):
        num_pos = 0
        num_impos = 0
        for i in labels[:, task]:
            if i == 1:
                num_pos = num_pos + 1
            if i == 0:
                num_impos = num_impos + 1
        weight = num_impos / (num_pos+0.00000001)
        task_pos_weight_list.append(weight)
    task_pos_weight = torch.tensor(task_pos_weight_list)
    return task_pos_weight

class EarlyStopping(object):
    def __init__(self, pretrained_model='Null_early_stop.pth', mode='higher', patience=10, filename=None, task_name="None"):
        if filename is None:
            task_name = task_name
            filename ='../Model_Save/{}_early_stop.pth'.format(task_name)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = pretrained_model

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)


    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        # model.load_state_dict(torch.load(self.filename)['model_state_dict'])
        model.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu'))['model_state_dict'])

    def load_pretrained_model(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked']
        if torch.cuda.is_available():
            pretrained_model = torch.load('../model/'+self.pretrained_model)
        else:
            pretrained_model = torch.load('../model/'+self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)

    def load_model_attention(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.bias',
                                 'weighted_sum_readout.shared_weighting.0.weight',
                                 'weighted_sum_readout.shared_weighting.0.bias',
                                 ]
        if torch.cuda.is_available():
            pretrained_model = torch.load('../model/' + self.pretrained_model)
        else:
            pretrained_model = torch.load('../model/' + self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))
        return scores

    def return_pred_true(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        return y_pred, y_true

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_squared_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.
        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            AUC_PRC for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred)
            scores.append(auc(recall, precision))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'mae', 'roc_prc', 'r2', 'return_pred_true'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", "mae", "roc_prc", "r2", "return_pred_true", got {}'.format(metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'roc_prc':
            return self.roc_precision_recall_score()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'return_pred_true':
            return self.return_pred_true()

def run_a_train_epoch_heterogeneous(args, epoch, model, data_loader, loss_criterion_c, loss_criterion_r, optimizer, task_weight=None):
    model.train()
    train_meter_c = Meter()
    train_meter_r = Meter()
    if task_weight is not None:
        task_weight = task_weight.float().to(args['device'])

    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, mask = batch_data
        mask = mask.float().to(args['device'])
        labels.float().to(args['device'])
        # print(labels.shape)
        atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
        logits = model(bg, atom_feats)
        # print(logits.shape)
        labels = labels.type_as(logits).to(args['device'])
        # calculate loss according to different task class
        if args['task_class'] == 'classification_regression':
            # split classification and regression
            logits_c = logits[:,:args['classification_num']]
            labels_c = labels[:,:args['classification_num']]
            mask_c = mask[:,:args['classification_num']]

            logits_r = logits[:,args['classification_num']:]
            labels_r = labels[:,args['classification_num']:]
            mask_r = mask[:,args['classification_num']:]
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_c(logits_c, labels_c)*(mask_c != 0).float()).mean() \
                       + (loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float()).mean()
            else:
                task_weight_c = task_weight[:args['classification_num']]
                task_weight_r = task_weight[args['classification_num']:]
                loss = (torch.mean(loss_criterion_c(logits_c, labels_c)*(mask_c != 0).float(), dim=0)*task_weight_c).mean() \
                       + (torch.mean(loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float(), dim=0)*task_weight_r).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter_c.update(logits_c, labels_c, mask_c)
            train_meter_r.update(logits_r, labels_r, mask_r)
            del bg, mask, labels, atom_feats,  loss, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
            torch.cuda.empty_cache()
        elif args['task_class'] == 'classification':
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_c(logits, labels)*(mask != 0).float()).mean()
            else:
                loss = (torch.mean(loss_criterion_c(logits, labels) * (mask != 0).float(),dim=0)*task_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter_c.update(logits, labels, mask)
            del bg, mask, labels, atom_feats, loss,  logits
            torch.cuda.empty_cache()
        else:
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_r(logits, labels)*(mask != 0).float()).mean()
            else:
                loss = (torch.mean(loss_criterion_r(logits, labels) * (mask != 0).float(), dim=0)*task_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter_r.update(logits, labels, mask)
            del bg, mask, labels, atom_feats,  loss,  logits
            torch.cuda.empty_cache()
    if args['task_class'] == 'classification_regression':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']) +
                              train_meter_r.compute_metric(args['regression_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], 'r2+auc', train_score))
    elif args['task_class'] == 'classification':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['classification_metric_name'], train_score))
    else:
        train_score = np.mean(train_meter_r.compute_metric(args['regression_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['regression_metric_name'], train_score))

def run_an_eval_epoch_heterogeneous_generate_weight(args, model, data_loader):
    model.eval()
    atom_list_all = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            print("batch: {}/{}".format(batch_id+1, len(data_loader)))
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            logits, atom_weight_list = model(bg, atom_feats)
            for atom_weight in atom_weight_list:
                atom_list_all.append(atom_weight[args['select_task_index']])
    task_name = args['select_task_list'][0]
    atom_weight_list = pd.DataFrame(atom_list_all, columns=['atom_weight'])
    atom_weight_list.to_csv(task_name+"_atom_weight.csv", index=None)


def generate_chemical_environment(args, model, data_loader):
    model.eval()
    atom_list_all = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            print("batch: {}/{}".format(batch_id + 1, len(data_loader)))
            smiles, bg, labels, mask = batch_data
            print(bg.ndata[args['atom_data_field']][1])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            logits, atom_weight_list = model(bg, atom_feats)
            print('after training:', bg.ndata['h'][1])

def run_an_eval_epoch_heterogeneous(args, model, data_loader):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(args['device'])
            mask = mask.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            logits = model(bg, atom_feats)
            labels = labels.type_as(logits).to(args['device'])
            if args['task_class'] == 'classification_regression':
                # split classification and regression
                logits_c = logits[:, :args['classification_num']]
                labels_c = labels[:, :args['classification_num']]
                mask_c = mask[:, :args['classification_num']]

                logits_r = logits[:, args['classification_num']:]
                labels_r = labels[:, args['classification_num']:]
                mask_r = mask[:, args['classification_num']:]
                # Mask non-existing labels
                eval_meter_c.update(logits_c, labels_c, mask_c)
                eval_meter_r.update(logits_r, labels_r, mask_r)
                del smiles, bg,  mask, labels, atom_feats, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
                torch.cuda.empty_cache()
            elif args['task_class'] == 'classification':
                # Mask non-existing labels
                eval_meter_c.update(logits, labels, mask)
                del smiles, bg,  mask, labels, atom_feats,  logits
                torch.cuda.empty_cache()
            else:
                # Mask non-existing labels
                eval_meter_r.update(logits, labels, mask)
                del smiles, bg,  mask, labels, atom_feats, logits
                torch.cuda.empty_cache()
        if args['task_class'] == 'classification_regression':
            return eval_meter_c.compute_metric(args['classification_metric_name']) + \
                   eval_meter_r.compute_metric(args['regression_metric_name'])
        elif args['task_class'] == 'classification':
            return eval_meter_c.compute_metric(args['classification_metric_name'])
        else:
            return eval_meter_r.compute_metric(args['regression_metric_name'])


