import os
import torch
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss, DerivLoss
import matplotlib.pyplot as plt
from utils.visual import visual
import numpy as np


class Exp_Steady(Exp_Basic):
    def __init__(self, args):
        super(Exp_Steady, self).__init__(args)

    def vali(self):
        myloss = L2Loss(size_average=False)
        self.model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for pos, fx, y in self.test_loader:
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x, fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)

                tl = myloss(out, y).item()
                rel_err += tl

        rel_err /= self.args.ntest
        return rel_err

    def train(self):
        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else: 
            raise ValueError('Optimizer only AdamW or Adam')
        
        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.args.pct_start)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        myloss = L2Loss(size_average=False)
        if self.args.derivloss:
            regloss = DerivLoss(size_average=False, shapelist=self.args.shapelist)

        for ep in range(self.args.epochs):

            self.model.train()
            train_loss = 0

            for pos, fx, y in self.train_loader:
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x, fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
                    y = self.dataset.y_normalizer.decode(y)

                if self.args.derivloss:
                    loss = myloss(out, y) + 0.1 * regloss(out, y)
                else:
                    loss = myloss(out, y)

                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()

                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                
                if self.args.scheduler == 'OneCycleLR':
                    scheduler.step()
            if self.args.scheduler == 'CosineAnnealingLR' or self.args.scheduler == 'StepLR':
                scheduler.step()

            train_loss = train_loss / self.args.ntrain
            print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))

            rel_err = self.vali()
            print("rel_err:{}".format(rel_err))

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save models')
                torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('final save models')
        torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

    def test(self):
        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt"))
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        rel_err = 0.0
        id = 0
        myloss = L2Loss(size_average=False)
        with torch.no_grad():
            for pos, fx, y in self.test_loader:
                id += 1
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x, fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
                tl = myloss(out, y).item()
                rel_err += tl
                if id < self.args.vis_num:
                    print('visual: ', id)
                    visual(x, y, out, self.args, id)

        rel_err /= self.args.ntest
        print("rel_err:{}".format(rel_err))
