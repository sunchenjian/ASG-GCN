import numpy as np
import torch

from config import *
from My_GNN import My_gnn
from dataloader import *
from utils import accuracy
import os
from sklearn import metrics
from sklearn.metrics import classification_report
import sys

sys.path.append("Compact_SER-main/")

if __name__ == '__main__':
    #传入参数
    args = OptInit().initialize()
    print('Loading dataset ...')

    label_dir = r'C:\Users\sun\Desktop\models\features\MODMA\all_label_MODMA.npy'
    feature_dir = r'C:\Users\sun\Desktop\models\features\MODMA\all_data_MODMA.npy'

    raw_features, y = load_data(label_dir, feature_dir)

    node_ftr = torch.Tensor(raw_features)

    n_folds = 10
    train_index, val_index, test_index = data_split(raw_features, y, n_folds)

    #情绪特征
    mood_model = torch.load("mood_model.pt")

    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros((n_folds))
    f1s = np.zeros((n_folds))
    pres = np.zeros((n_folds))
    recs = np.zeros((n_folds))

    train_num = 0
    val_num = 0
    test_num = 0

    filename = '.\log\log_DAIC.txt'
    log = Logger(filename)
    sys.stdout = log

    for fold in range(n_folds):
        print("\r\n============================== Fold {} =================================".format(fold))
        train_ind = train_index[fold]
        val_ind = val_index[fold]
        test_ind = test_index[fold]
        val_num += len(val_ind)
        test_num += len(test_ind)

        edge_index = get_A(node_ftr.shape[1], 10, 10)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(args.device)

        model = My_gnn(node_ftr.shape[2], args.num_classes, args.dropout, edge_dropout=args.edropout, hgc=args.hgc,
                       lg=args.lg, device=args.device).to(args.device)
        model = model.to(args.device)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(args.device)
        labels = torch.tensor(y, dtype=torch.long).to(args.device)
        fold_model_path = args.ckpt_path + '/fold{}.pth'.format(fold)


        def train():
            print("Number of training samples %d" % len(train_ind))
            print("Start training ....\r\n")
            acc = 0
            for epoch in range(args.num_iter):
                mood_feature = mood_model(node_ftr.to(args.device))
                model.train()
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    node_logits = model(features_cuda, mood_feature, edge_index)    ## , edge_index, edgenet_input
                    loss = loss_fn(node_logits[train_ind], labels[train_ind])
                    # print(node_logits[train_ind])
                    loss.backward()
                    optimizer.step()
                logits_train = node_logits[train_ind].detach().cpu().numpy()

                correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])
                pre1, rec1, _, _ = metrics.precision_recall_fscore_support(y[train_ind],
                                                                         np.argmax(logits_train, axis=1),
                                                                         average='binary')

                model.eval()
                with torch.set_grad_enabled(False):
                    node_logits = model(features_cuda, mood_feature, edge_index)
                    val_loss = loss_fn(node_logits[val_ind], labels[val_ind])
                logits_val = node_logits[val_ind].detach().cpu().numpy()
                correct_test, acc_test = accuracy(logits_val, y[val_ind])
                pre, rec, _, _ = metrics.precision_recall_fscore_support(y[val_ind],
                                                                         np.argmax(logits_val, axis=1), average='binary')

                print("Epoch: {},\tce loss: {:.5f},\ttrain_acc: {:.5f},\tpre: {:.5f},\trec: {:.5f}".format(epoch, loss.item(), acc_train.item(), pre1, rec1))
                print("\t\t\tval_loss: {:.5f},\tval_acc: {:.5f},\tpre: {:.5f},\trec: {:.5f}\n".format(val_loss.item(), acc_test.item(), pre, rec))
                if acc_test > acc and epoch > 9:
                    acc = acc_test

                    if not os.path.exists(args.ckpt_path):
                        os.makedirs(args.ckpt_path)

                    torch.save(model.state_dict(), fold_model_path)


        def evaluate():
            mood_feature = mood_model(node_ftr.to(args.device))
            print("Number of testing samples %d" % len(test_ind))
            print("Start testing ...")
            model.load_state_dict(torch.load(fold_model_path))
            model.eval()
            node_logits = model(features_cuda, mood_feature, edge_index)
            logits_test = node_logits[test_ind].detach().cpu().numpy()
            pre, rec, f1, _ = metrics.precision_recall_fscore_support(y[test_ind],
                                                                     np.argmax(logits_test, axis=1), average='binary')

            print(classification_report(y[test_ind], np.argmax(logits_test, axis=1)))
            corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
            f1s[fold] = f1
            pres[fold] = pre
            recs[fold] = rec
            print("Fold {} test accuracy {:.5f}".format(fold, accs[fold]))
            print("\r\n => Fold {}  test f1 {:5f}, precision {:5f}, recall {:5f}".format(fold, f1s[fold], pres[fold], recs[fold]))

        if args.train == 1:
            train()
        # elif args.train == 0:
            evaluate()


    print("\r\n============================Finish===============================")
    n_samples = test_num
    acc_nfold = np.sum(corrects)/ n_samples
    print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
    print("=> Average test f1 {:5f}, precision {:5f}, recall {:5f} in {}-fold CV.".format(np.mean(f1s), np.mean(pres),
                                                                                          np.mean(recs), n_folds))


