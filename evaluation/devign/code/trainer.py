import copy
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rc, f1_score as f1

from tqdm import tqdm
import json
from utils import debug


def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            names, graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), f1_score(all_targets, all_predictions) * 100
    pass


def get_embeddings(model, loss_function, num_batches, data_iter, after_ggnn_file):
    model.eval()
    after_ggnn = []
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            names, graph, targets = data_iter()
            targets = targets.cuda()
            embeddings = []
            predictions = model(graph, cuda=True, embeddings=embeddings)
            # import pdb
            # pdb.set_trace()
            for iii, embedding in enumerate(embeddings[0]):
                obj = {}
                # obj["name"] = names[iii]
                obj["target"] = int(targets[iii].tolist())
                obj["graph_feature"] = embedding
                after_ggnn.append(obj)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        f = open(after_ggnn_file, "w")
        json.dump(after_ggnn, f)
        f.close()

        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


def get_corrects(model, loss_function, num_batches, data_iter, correct_file):
    model.eval()
    after_ggnn = []
    correct_names = []
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        all_names = []
        for _ in range(num_batches):
            names, graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            '''
            for iii,embedding in enumerate(embeddings[0]):
                obj={}
                obj["name"] = names[iii]
                obj["target"] = int(targets[iii].tolist())
                obj["graph_feature"] = embedding
                after_ggnn.append(obj)
            '''
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            all_names.extend(names)
        for iii in range(len(all_names)):
            if int(all_targets[iii]) == int(all_predictions[iii]):
                correct_names.append(all_names[iii])

        model.train()
        f = open(correct_file, "w")
        json.dump(correct_names, f, indent=4)
        f.close()

        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


def evaluate_metrics(model, loss_function, num_batches, data_iter, neg_metrics=[], avg_metrics=[], other_metrics=[]):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            names, graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            # import pdb
            # pdb.set_trace()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i, prediction in enumerate(all_predictions):
            if prediction == 1 and all_targets[i] == 1:
                tp += 1
            if prediction == 1 and all_targets[i] == 0:
                fp += 1
            if prediction == 0 and all_targets[i] == 0:
                tn += 1
            if prediction == 0 and all_targets[i] == 1:
                fn += 1
        # neg_acc = (tn+tp)/(tn+tp+fn+fp)*100
        # neg_prec = (tn)/(tn+fn)*100
        # neg_recall = (tn)/(tn+fp)*100
        # neg_f1 = 2*neg_prec*neg_recall/(neg_prec+neg_recall)

        pos_acc = accuracy_score(all_targets, all_predictions) * 100
        pos_prec = precision_score(all_targets, all_predictions) * 100
        pos_recall = recall_score(all_targets, all_predictions) * 100
        pos_f1 = f1_score(all_targets, all_predictions) * 100

        neg_acc = accuracy_score(all_targets, all_predictions) * 100
        neg_prec = precision_score(all_targets, all_predictions, pos_label=0) * 100
        neg_recall = recall_score(all_targets, all_predictions, pos_label=0) * 100
        neg_f1 = f1_score(all_targets, all_predictions, pos_label=0) * 100

        avg_acc = accuracy_score(all_targets, all_predictions) * 100
        avg_prec = precision_score(all_targets, all_predictions, average='macro') * 100
        avg_recall = recall_score(all_targets, all_predictions, average='macro') * 100
        avg_f1 = f1_score(all_targets, all_predictions, average='macro') * 100

        from sklearn.metrics import roc_auc_score, matthews_corrcoef
        auc = roc_auc_score(all_targets, all_predictions)
        mcc = matthews_corrcoef(all_targets, all_predictions)

        from math import sqrt
        pos_g_measure = sqrt(pos_prec * pos_recall)
        neg_g_measure = sqrt(neg_prec * neg_recall)
        avg_g_measure = sqrt(avg_prec * avg_recall)

        neg_metrics.extend([neg_acc, neg_prec, neg_recall, neg_f1])
        avg_metrics.extend([avg_acc, avg_prec, avg_recall, avg_f1])
        other_metrics.extend([auc, mcc, pos_g_measure, neg_g_measure, avg_g_measure])

        return pos_acc, \
            pos_prec, \
            pos_recall, \
            pos_f1










def train(model, dataset, max_steps, dev_every, loss_function, optimizer, save_path, log_every, max_patience,
          train_mode):
    return train_epochs(model, dataset, 50, 1, loss_function, optimizer, save_path, log_every, max_patience)



def train_epochs(model, dataset, epochs, dev_every, loss_function, optimizer, save_path, log_every=50,
                 max_patience=5):
    debug('Start Training')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0

    class_weights = torch.tensor([1.0, 1.0]).cuda()
    debug('Weights : %s' % class_weights)

    try:
        for epoch in range(epochs):
            model.train()  # Set model to training mode
            epoch_loss = 0  # Track loss for the epoch
            # import pdb
            # pdb.set_trace()
            for step_count in tqdm(range(len(dataset.train_examples) // dataset.batch_size)):  # Iterate over batches in the epoch
                model.zero_grad()
                names, graph, targets = dataset.get_next_train_batch()
                targets = targets.cuda()

                # Forward pass
                predictions = model(graph, cuda=True)

                # Calculate loss
                weights = class_weights[targets.long()]
                weighted_targets = targets * weights
                batch_loss = loss_function(predictions, weighted_targets)
                epoch_loss += batch_loss.item()  # Accumulate loss for the epoch

                # Backward pass and optimization
                batch_loss.backward()
                optimizer.step()

                # # Logging
                # if log_every is not None and (step_count % log_every == log_every - 1):
                #     debug('Epoch %d, Step %d\t\tTrain Loss %10.3f' % (
                #     epoch, step_count, batch_loss.detach().cpu().item()))

            # End of epoch, log the loss for the epoch
            debug('Epoch %d: Average Train Loss %10.3f' % (epoch, epoch_loss / (len(dataset.train_examples) // dataset.batch_size)))

            # Every epoch, evaluate the model
            if epoch % dev_every == dev_every - 1:
                # Evaluate on test data (or validation)
                acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test1_batch(),
                                                   dataset.get_next_test1_batch)
                debug('Epoch %d: Test1 Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (
                epoch, acc, pr, rc, f1))

                # Save model at the end of each epoch
                _save_file = open(save_path + '-model_epoch_' + str(epoch), 'wb')
                torch.save(model, _save_file)
                _save_file.close()

                # Early stopping logic (optional)
                # You can include patience logic here if needed

            # Early stopping based on patience (optional)
            # if patience_counter == max_patience:
            #     debug('Early stopping due to no improvement in F1 score')
            #     break

    except KeyboardInterrupt:
        debug('Training Interrupted by user!')

    # Save final model after training
    _save_file = open(save_path + '-model_final.bin', 'wb')
    torch.save(model, _save_file)
    _save_file.close()

    debug('=' * 100)
    return model