import numpy as np
import torch
from torch.utils import data
from CNN_model import ConvNet
from dataloader import MelSpecDataset, split_ids
from sklearn import metrics
import pickle


class Solver(object):
    def __init__(self, epochs=2, lr=0.0001, split_r=0.3, batch_size=128,
                 tr_test_state='', resume_training=False, tag_translate=[],
                 save_checkpoint_path='', save_best_model_path='', data_dir='',
                 test_log_path='', debug=False):

        assert tr_test_state in ['train', 'test']
        self.batch_size = batch_size
        self.tr_test_state = tr_test_state
        self.is_cuda = torch.cuda.is_available()
        self.save_best_model_path = save_best_model_path
        self.data_dir = data_dir
        self.debug = debug
        self.lr = lr
        self.epochs = epochs
        self.split_r = split_r
        self.resume_training = resume_training
        self.save_checkpoint_path = save_checkpoint_path

        self._build_model()

        if self.tr_test_state == 'train':  # this nested if may be bad
            if self.resume_training:
                self._load_checkpoint()
            else:
                tr_split, val_split = split_ids(dir=self.data_dir, split_r=self.split_r)
                self.train_split = tr_split
                self.validation_split = val_split
                self.save_epoch = 0
                self.t_loss = []
                self.v_loss = []
                self.best_stat = 0

            self._fetch_tr_loaders()
            self.model.train()

        if self.tr_test_state == 'test':
            self.test_log_path = test_log_path
            m = torch.load(self.save_best_model_path)
            self.model.load_state_dict(m['model'])
            self._fetch_test_loader()
            self.model.eval()

    def _build_model(self):
        model = ConvNet(num_class=1000)
        self.model = model
        if self.is_cuda:
            self.model.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.optimizer = optimizer

    def _fetch_tr_loaders(self):
        full_set = MelSpecDataset(dir=self.data_dir)
        train_set = data.Subset(full_set, self.train_split)
        validation_set = data.Subset(full_set, self.validation_split)
        train_set = self._to_debug(train_set)
        validation_set = self._to_debug(validation_set)
        tr_loader = data.DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size)
        val_loader = data.DataLoader(dataset=validation_set, shuffle=True, batch_size=self.batch_size)

        self.data_loader = tr_loader
        self.validation_loader = val_loader

    def _fetch_test_loader(self):
        test_set = MelSpecDataset(dir=self.data_dir)
        test_set = self._to_debug(test_set)
        test_loader = data.DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size)

        self.data_loader = test_loader

    def _to_debug(self, dataset):
        debug_size = 5
        if self.debug:
            dataset = data.Subset(dataset, np.arange(debug_size*self.batch_size))

        return dataset

    def _load_checkpoint(self):
        checkpoint = torch.load(self.save_checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.save_epoch = checkpoint['epoch']
        self.v_loss = checkpoint['v_loss']
        self.t_loss = checkpoint['t_loss']
        self.train_split = checkpoint['tr_split']
        self.validation_split = checkpoint['val_split']
        self.best_stat = checkpoint['best_stat']

        print(f'loaded checkpoint. Next ep: {self.save_epoch}')

    def _save_checkpoint(self, curr_epoch):
        torch.save({
            'epoch': curr_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            't_loss': self.t_loss,
            'v_loss': self.v_loss,
            'tr_split': self.train_split,
            'val_split': self.validation_split,
            'best_stat': self.best_stat
            }, self.save_checkpoint_path + f"EP{curr_epoch}")

    def _to_cuda(self, x):
        if self.is_cuda:
            x = x.cuda()
        return x

    def train(self):
        assert self.tr_test_state == 'train'

        reconst_loss = torch.nn.BCELoss()
        for epoch in range(self.save_epoch, self.epochs):
            self.model.train()
            running_loss = 0
            b = 0
            for x, y in iter(self.data_loader):
                x = self._to_cuda(x)
                y = self._to_cuda(y)

                # predict
                out = self.model(x)
                loss = reconst_loss(out, y)

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                b += 1
                print(f'tr epoch {epoch}/{self.epochs-1}, batch {b}/{len(self.data_loader)}')

            ep_loss = running_loss/len(self.data_loader)
            print(f'loss {ep_loss}')
            self.t_loss.append(ep_loss)

            if epoch % 5 == 0:
                print(f'in val + save clause, ep: {epoch}')
                _, _, roc, _, _ = self._validation()

                if roc >= self.best_stat:
                    self.best_stat = roc
                    m = self.model.state_dict()
                    torch.save({'model': m}, self.save_best_model_path)

                self._save_checkpoint(curr_epoch=epoch)

    def _validation(self):
        prd_array = []  # prediction
        gt_array = []   # ground truth
        self.model.eval()
        reconst_loss = torch.nn.BCELoss()

        running_loss = 0
        b = 0
        for x, y in iter(self.validation_loader):
            # variables to cuda
            x = self._to_cuda(x)
            y = self._to_cuda(y)

            # predict
            out = self.model(x)
            loss = reconst_loss(out, y)

            out = out.detach().cpu()
            y = y.detach().cpu()

            running_loss += loss.item()
            b += 1
            print(f'val batch {b}/{len(self.validation_loader)}')

            for prd in out:
                prd_array.append(list(np.array(prd)))
            for gt in y:
                gt_array.append(list(np.array(gt)))

        v_loss = running_loss/len(self.validation_loader)
        print(f'loss {v_loss}')
        self.v_loss.append(v_loss)

        pr_auc_ma, pr_auc_mi, roc_auc_ma, roc_auc_mi, acc = self._calc_stats(prd_array, gt_array)

        return pr_auc_ma, pr_auc_mi, roc_auc_ma, roc_auc_mi, acc

    def _calc_stats(self, prd_array, gt_array):
        prd_array = np.array(prd_array)
        gt_array = np.array(gt_array)

        roc_auc_ma = 0
        roc_auc_mi = 0
        pr_auc_ma = 0
        pr_auc_mi = 0

        if not self.debug:
            roc_auc_ma = metrics.roc_auc_score(gt_array, prd_array, average='macro')
            roc_auc_mi = metrics.roc_auc_score(gt_array, prd_array, average='micro')

            pr_auc_ma = metrics.average_precision_score(gt_array, prd_array, average='macro')
            pr_auc_mi = metrics.average_precision_score(gt_array, prd_array, average='micro')

        prd_cls = prd_array > 0.5
        prd_cls.astype('float32')
        acc = metrics.accuracy_score(gt_array, prd_cls)

        return pr_auc_ma, pr_auc_mi, roc_auc_ma, roc_auc_mi, acc

    def test(self):
        assert self.tr_test_state == 'test', "create a new solver for testing"

        reconst_loss = torch.nn.BCELoss()
        prd_array = []  # prediction
        gt_array = []   # ground truth

        running_loss = 0
        b = 0
        for x, y in iter(self.data_loader):
            # variables to cuda
            x = self._to_cuda(x)
            y = self._to_cuda(y)

            # predict
            out = self.model(x)
            loss = reconst_loss(out, y)

            running_loss += loss.item()
            b += 1
            print(f'test batch {b}/{len(self.data_loader)}')

            # append prediction
            out = out.detach().cpu()
            y = y.detach().cpu()
            for prd in out:
                prd_array.append(list(np.array(prd)))
            for gt in y:
                gt_array.append(list(np.array(gt)))

        print(f'loss {running_loss/len(self.data_loader)}')

        pr_auc_ma, pr_auc_mi, roc_auc_ma, roc_auc_mi, acc = self._calc_stats(prd_array, gt_array)

        # write to file
        log_file = self.test_log_path + 'log.txt'
        prd_file = self.test_log_path + 'prd.npy'
        gt_file = self.test_log_path + 'gt.npy'

        with open(log_file, 'w') as f:
            f.write(str(pr_auc_ma) + '\n')
            f.write(str(pr_auc_mi) + '\n')
            f.write(str(roc_auc_ma) + '\n')
            f.write(str(roc_auc_mi) + '\n')
            f.write(str(acc) + '\n')

        with open(prd_file, 'wb') as f:
            np.save(f, prd_array)

        with open(gt_file, 'wb') as f:
            np.save(f, gt_array)


def train(checkpoint_path, model_path, data_path, debug=True):
    tr_solver = Solver(epochs=1, lr=0.0001, split_r=0.3, batch_size=128,
                       tr_test_state='train', resume_training=False,
                       tag_translate=[], save_checkpoint_path=checkpoint_path,
                       save_best_model_path=model_path, data_dir=data_path,
                       debug=debug)
    tr_solver.train()


def cont_train(checkpoint_path, model_path, data_path, debug=True):
    tr_solver = Solver(epochs=2, lr=0.0001, split_r=0.3, batch_size=128,
                       tr_test_state='train', resume_training=True,
                       tag_translate=[], save_checkpoint_path=checkpoint_path,
                       save_best_model_path=model_path, data_dir=data_path,
                       debug=debug)
    tr_solver.train()


def test(model_path, data_path, test_log_path, debug=True):
    test_solver = Solver(batch_size=128,
                         tr_test_state='test',
                         tag_translate=[],
                         save_best_model_path=model_path, data_dir=data_path,
                         test_log_path=test_log_path, debug=debug)
    test_solver.test()


if __name__ == '__main__':
    # intended use

    checkpoint_path = '/Users/karlsimu/Programming/Python/School/ID2223/proj/checkpoints/'
    model_path = '/Users/karlsimu/Programming/Python/School/ID2223/proj/best_model/model'
    train_data_path = '/Users/karlsimu/Programming/Datasets/spec_tags_top_1000'
    test_data_path = '/Users/karlsimu/Programming/Datasets/spec_tags_top_1000_val'
    test_log_path = 'logs/'

    # train(checkpoint_path=checkpoint_path, model_path=model_path,
    #       data_path=train_data_path, debug=True)

    # cont_train(checkpoint_path=checkpoint_path+'EP0', model_path=model_path,
    #            data_path=train_data_path, debug=True)

    # test(model_path=model_path, data_path=test_data_path,
    #      test_log_path=test_log_path, debug=True)
