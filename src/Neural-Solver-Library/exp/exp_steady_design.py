import os
import time
import torch
import torch.nn as nn
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss, DerivLoss
import matplotlib.pyplot as plt
from utils.visual import visual
from utils.drag_coefficient import cal_coefficient
import numpy as np
import scipy as sc


class Exp_Steady_Design(Exp_Basic):
    def __init__(self, args):
        super(Exp_Steady_Design, self).__init__(args)

    def vali(self):
        myloss = nn.MSELoss(reduction='none')
        self.model.eval()
        rel_err = 0.0
        index = 0
        with torch.no_grad():
            for pos, fx, y, surf, geo, obj_file in self.test_loader:
                x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x.unsqueeze(0), fx.unsqueeze(0), geo=geo)[0]
                loss_press = myloss(out[surf, -1], y[surf, -1]).mean(dim=0)
                loss_velo_var = myloss(out[:, :-1], y[:, :-1]).mean(dim=0)
                loss_velo = loss_velo_var.mean()
                loss = loss_velo + 0.5 * loss_press
                rel_err += loss.item()
                index += 1

        rel_err /= float(index)
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
        myloss = nn.MSELoss(reduction='none')

        for ep in range(self.args.epochs):

            self.model.train()
            train_loss = 0
            index = 0
            for pos, fx, y, surf, geo in self.train_loader:
                x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x.unsqueeze(0), fx.unsqueeze(0), geo=geo)[0]
                loss_press = myloss(out[surf, -1], y[surf, -1]).mean(dim=0)
                loss_velo_var = myloss(out[:, :-1], y[:, :-1]).mean(dim=0)
                loss_velo = loss_velo_var.mean()
                loss = loss_velo + 0.5 * loss_press

                train_loss += loss.item()
                index += 1
                optimizer.zero_grad()
                loss.backward()

                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                if self.args.scheduler == 'OneCycleLR':
                    scheduler.step()
            if self.args.scheduler == 'CosineAnnealingLR' or self.args.scheduler == 'StepLR':
                scheduler.step()

            train_loss = train_loss / float(index)
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

        criterion_func = nn.MSELoss(reduction='none')
        l2errs_press = []
        l2errs_velo = []
        mses_press = []
        mses_velo_var = []
        times = []
        gt_coef_list = []
        pred_coef_list = []
        coef_error = 0
        index = 0
        with torch.no_grad():
            for pos, fx, y, surf, geo, obj_file in self.test_loader:
                x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                tic = time.time()
                out = self.model(x.unsqueeze(0), fx.unsqueeze(0), geo=geo)[0]
                toc = time.time()

                if self.test_loader.coef_norm is not None:
                    mean = torch.tensor(self.test_loader.coef_norm[2]).cuda()
                    std = torch.tensor(self.test_loader.coef_norm[3]).cuda()
                    pred_press = out[surf, -1] * std[-1] + mean[-1]
                    gt_press = y[surf, -1] * std[-1] + mean[-1]
                    pred_surf_velo = out[surf, :-1] * std[:-1] + mean[:-1]
                    gt_surf_velo = y[surf, :-1] * std[:-1] + mean[:-1]
                    pred_velo = out[~surf, :-1] * std[:-1] + mean[:-1]
                    gt_velo = y[~surf, :-1] * std[:-1] + mean[:-1]

                pred_coef = cal_coefficient(obj_file.split('/')[1], pred_press[:, None].detach().cpu().numpy(),
                                            pred_surf_velo.detach().cpu().numpy())
                gt_coef = cal_coefficient(obj_file.split('/')[1], gt_press[:, None].detach().cpu().numpy(),
                                          gt_surf_velo.detach().cpu().numpy())

                gt_coef_list.append(gt_coef)
                pred_coef_list.append(pred_coef)
                coef_error += (abs(pred_coef - gt_coef) / gt_coef)

                l2err_press = torch.norm(pred_press - gt_press) / torch.norm(gt_press)
                l2err_velo = torch.norm(pred_velo - gt_velo) / torch.norm(gt_velo)

                mse_press = criterion_func(out[surf, -1], y[surf, -1]).mean(dim=0)
                mse_velo_var = criterion_func(out[~surf, :-1], y[~surf, :-1]).mean(dim=0)

                l2errs_press.append(l2err_press.cpu().numpy())
                l2errs_velo.append(l2err_velo.cpu().numpy())
                mses_press.append(mse_press.cpu().numpy())
                mses_velo_var.append(mse_velo_var.cpu().numpy())
                times.append(toc - tic)
                index += 1

        gt_coef_list = np.array(gt_coef_list)
        pred_coef_list = np.array(pred_coef_list)
        spear = sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]
        print("rho_d: ", spear)
        print("c_d: ", coef_error / index)
        l2err_press = np.mean(l2errs_press)
        l2err_velo = np.mean(l2errs_velo)
        rmse_press = np.sqrt(np.mean(mses_press))
        rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))
        if self.test_loader.coef_norm is not None:
            rmse_press *= self.test_loader.coef_norm[3][-1]
            rmse_velo_var *= self.test_loader.coef_norm[3][:-1]
        print('relative l2 error press:', l2err_press)
        print('relative l2 error velo:', l2err_velo)
        print('press:', rmse_press)
        print('velo:', rmse_velo_var, np.sqrt(np.mean(np.square(rmse_velo_var))))
        print('time:', np.mean(times))
