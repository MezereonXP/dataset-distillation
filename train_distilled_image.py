import time
import os
import logging
import random
import math
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import networks
from itertools import repeat, chain
from networks.utils import clone_tuple
from utils.distributed import broadcast_coalesced, all_reduce_coalesced
from utils.io import save_results
from utils.label_inits import distillation_label_initialiser
from basics import task_loss, final_objective_loss, evaluate_steps
from contextlib import contextmanager
torch.backends.cudnn.enabled=False
#import faulthandler
#faulthandler.enable()

class Trainer(object):
    def __init__(self, state, models):
        self.state = state
        self.models = models
        self.num_data_steps = state.distill_steps  # how much data we have
        self.T = state.distill_steps * state.distill_epochs  # how many sc steps we run
        self.num_per_step = state.num_distill_classes * state.distilled_images_per_class_per_step
        assert state.distill_lr >= 0, 'distill_lr must >= 0'
        assert len(state.init_labels) == state.num_distill_classes, 'len(init_labels) must == num_distill_classes'
        self.init_data_optim()

    def init_data_optim(self):
        self.params = []
        state = self.state
        req_lbl_grad = not state.static_labels
        # labels
        self.labels = []

        # num_data_steps = distill_steps (default: 10)
        for _ in range(self.num_data_steps):  # Generate the label vector and Append these label to the variables [params and label]
            if state.random_init_labels:
                distill_label = distillation_label_initialiser(state, self.num_per_step, torch.float, req_lbl_grad)
            else:
                # distilled_images_per_class_per_step = 1 [default]
                # Generate one-hot label
                if state.num_classes == 2:
                    dl_array = [[i == j for i in range(1)]for j in state.init_labels] * state.distilled_images_per_class_per_step
                else:
                    dl_array = [[i == j for i in range(state.num_classes)]for j in state.init_labels] * state.distilled_images_per_class_per_step

                distill_label = torch.tensor(dl_array,
                                             dtype=torch.float,
                                             requires_grad=req_lbl_grad,
                                             device=state.device)

            self.labels.append(distill_label)
            if not state.static_labels:
                self.params.append(distill_label)

        self.all_labels = torch.cat(self.labels)
        self.data = []
        for _ in range(self.num_data_steps):  # Init the synthetic data
            # num_per_step is the number of total generated images in a step
            # num_per_step = num_classes * distilled_images_per_class_per_step
            if state.textdata:
                distill_data = torch.randn(self.num_per_step, state.nc, state.input_size, state.ninp,
                                           device=state.device, requires_grad=(not state.freeze_data))
            else:
                # nc is channels number
                # cifar10 data likes [30, 3, 32, 32]
                distill_data = torch.randn(self.num_per_step, state.nc, state.input_size, state.input_size,
                                           device=state.device, requires_grad=(not state.freeze_data))

            self.data.append(distill_data)
            if not state.freeze_data:
                self.params.append(distill_data)

        # lr

        # undo the softplus + threshold
        # default is 0.02
        raw_init_distill_lr = torch.tensor(state.distill_lr, device=state.device)
        raw_init_distill_lr = raw_init_distill_lr.repeat(self.T, 1)
        self.raw_distill_lrs = raw_init_distill_lr.expm1_().log_().requires_grad_()
        self.params.append(self.raw_distill_lrs)

        assert len(self.params) > 0, "must have at least 1 parameter"

        # now all the params are in self.params, sync if using distributed
        if state.distributed:
            broadcast_coalesced(self.params)
            logging.info("parameters broadcast done!")

        self.optimizer = optim.Adam(self.params, lr=state.lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=state.decay_epochs,
                                                   gamma=state.decay_factor)
        for p in self.params:
            p.grad = torch.zeros_like(p)

    def get_steps(self):
        # 相当于把每一个step的数据复制epoch次，然后平铺
        data_label_iterable = (x for _ in range(self.state.distill_epochs) for x in zip(self.data, self.labels))
        lrs = F.softplus(self.raw_distill_lrs).unbind()
        steps = []
        for (data, label), lr in zip(data_label_iterable, lrs):
            steps.append((data, label, lr))
        # 包含 self.T 个数据，也就是 每个step包含的生成图片的数量 * epoch数
        return steps

    def forward(self, model, rdata, rlabel, steps):
        state = self.state

        # forward
        model.train()
        w = model.get_param()
        params = [w]
        gws = []
        # 先使用生成数据进行训练，这里没有使用优化器，手动进行梯度下降
        for step_i, (data, label, lr) in enumerate(steps):
            with torch.enable_grad():
                model.distilling_flag = True
                output = model.forward_with_param(data, w)
                loss = task_loss(state, output, label)
                lr = lr.squeeze()

            # 计算网络梯度，并且将历史梯度保存起来
            gw, = torch.autograd.grad(loss, w, lr, create_graph=True)

            with torch.no_grad():
                # SGD Update the model
                new_w = w.sub(gw).requires_grad_()
                # 把历史的模型权重保存起来
                params.append(new_w)
                gws.append(gw)
                w = new_w

        # final L
        model.eval()
        model.distilling_flag = False
        # 使用真实数据作为输入，得到输出，并且计算交叉熵误差
        output = model.forward_with_param(rdata, params[-1])
        ll = final_objective_loss(state, output, rlabel)  # cross-entropy loss for multi-classes
        print(f"params's length is {len(params)}")
        print(f"gws's length is {len(gws)}")
        return ll, (ll, params, gws)

    def backward(self, model, rdata, rlabel, steps, saved_for_backward):
        l, params, gws = saved_for_backward
        state = self.state

        datas = []
        gdatas = []
        lrs = []
        glrs = []
        labels = []
        glabels = []
        if state.textdata:
            with torch.backends.cudnn.flags(enabled=False):
                dw, = torch.autograd.grad(l, (params[-1],), retain_graph=True)
        else:
            # 计算真实数据下的交叉熵误差得到的网络梯度
            dw, = torch.autograd.grad(l, (params[-1],), retain_graph=True)

        # backward
        model.train()
        # Notation:
        #   math:    \grad is \nabla
        #   symbol:  d* means the gradient of final L w.r.t. *
        #            dw is \d L / \dw
        #            dgw is \d L / \d (\grad_w_t L_t )
        # We fold lr as part of the input to the step-wise loss
        #
        #   gw_t     = \grad_w_t L_t       (1)
        #   w_{t+1}  = w_t - gw_t          (2)
        #
        # Invariants at beginning of each iteration:
        #   ws are BEFORE applying gradient descent in this step
        #   Gradients dw is w.r.t. the updated ws AFTER this step
        #      dw = \d L / d w_{t+1}
        # !这里有一个坑：由于params和gws的长度不一致，所以zip会把params的最后一个丢弃
        for (data, label, lr), w, gw in reversed(list(zip(steps, params[:-1], gws))):
            # hvp_in are the tensors we need gradients w.r.t. final L:
            #   lr (if learning)
            #   data
            #   ws (PRE-GD) (needed for next step)
            #
            # source of gradients can be from:
            #   gw, the gradient in this step, whose gradients come from:
            #     the POST-GD updated ws
            # 实际上 hvp_in 就是我们想要更新一些个可学习的变量：网络参数，学习率，生成数据，标签
            hvp_in = [w]
            if not state.freeze_data:
                hvp_in.append(data)
            hvp_in.append(lr)
            if not state.static_labels:
                hvp_in.append(label)

            dgw = dw.neg()  # gw is already weighted by lr, so simple negation
            # hvp_grad 类似于二阶导数，乘上一个负的一阶导数dgw ???
            hvp_grad = torch.autograd.grad(
                outputs=(gw,),
                inputs=hvp_in,
                grad_outputs=(dgw,),
                retain_graph=True
            )
            # Update for next iteration, i.e., previous step
            with torch.no_grad():
                # Save the computed gdata and glrs
                if not state.freeze_data:
                    datas.append(data)  # 保存生成数据
                    gdatas.append(hvp_grad[1])  # 保存生成数据的梯度
                    lrs.append(lr)  # 保存学习率
                    glrs.append(hvp_grad[2])  # 保存学习率的梯度
                    if not state.static_labels:  # 如果标签可以学习，那么保存标签和标签的梯度
                        labels.append(label)
                        glabels.append(hvp_grad[3])
                else:
                    lrs.append(lr)
                    glrs.append(hvp_grad[1])
                    if not state.static_labels:
                        labels.append(label)
                        glabels.append(hvp_grad[2])

                # Update for next iteration, i.e., previous step
                # Update dw
                # dw becomes the gradients w.r.t. the updated w for previous step
                dw.add_(hvp_grad[0])

        return datas, gdatas, lrs, glrs, labels, glabels

    def accumulate_grad(self, grad_infos):
        bwd_out = []
        bwd_grad = []
        for datas, gdatas, lrs, glrs, labels, glabels in grad_infos:
            bwd_out += list(lrs)
            bwd_grad += list(glrs)
            if not self.state.freeze_data:
                for d, g in zip(datas, gdatas):
                    d.grad.add_(g)
            if not self.state.static_labels:
                for l, g in zip(labels, glabels):
                    l.grad.add_(g)
        if len(bwd_out) > 0:
            # 反向传播，方便优化器进行优化
            torch.autograd.backward(bwd_out, bwd_grad)  # MULTISTEP PROBLEM?

    def save_results(self, steps=None, visualize=True, subfolder=''):
        with torch.no_grad():
            steps = steps or self.get_steps()
            save_results(self.state, steps, visualize=visualize, subfolder=subfolder)

    def __call__(self):
        return self.train()

    def prefetch_train_loader_iter(self):
        state = self.state
        # device = state.device
        train_iter = iter(state.train_loader)
        if state.textdata:
            niter = len(tuple(train_iter))
        else:
            niter = len(train_iter)
        for epoch in range(state.epochs):
            if state.textdata:
                train_iter = iter(state.train_loader)
            print("Training Epoch: {}".format(epoch))
            prefetch_it = max(0, niter - 2)
            for it, example in enumerate(train_iter):
                if state.textdata:
                    data = example.text[0]
                    target = example.label
                    val = (data, target)
                else:
                    val = example
                
                # Prefetch (start workers) at the end of epoch BEFORE yielding
                if it == prefetch_it and epoch < state.epochs - 1:
                    train_iter = iter(state.train_loader)
                yield epoch, it, val

    def train(self):
        state = self.state
        device = state.device
        train_loader = state.train_loader
        sample_n_nets = state.local_sample_n_nets
        grad_divisor = state.sample_n_nets  # i.e., global sample_n_nets
        ckpt_int = state.checkpoint_interval

        data_t0 = time.time()

        for epoch, it, (rdata, rlabel) in self.prefetch_train_loader_iter():
            data_t = time.time() - data_t0
            
            
            if it == 0:
                self.scheduler.step()

            if it == 0 and ((ckpt_int >= 0 and epoch % ckpt_int == 0) or epoch == 0):
                with torch.no_grad():
                    steps = self.get_steps()
                self.save_results(steps=steps, visualize=state.visualize, subfolder='checkpoints/epoch{:04d}'.format(epoch))
                evaluate_steps(state, steps, 'Begin of epoch {}'.format(epoch))
            
            do_log_this_iter = it == 0 or (state.log_interval >= 0 and it % state.log_interval == 0)
            
            self.optimizer.zero_grad()
            rdata, rlabel = rdata.to(device, non_blocking=True), rlabel.to(device, non_blocking=True)

            if sample_n_nets == state.local_n_nets:
                tmodels = self.models
            else:
                idxs = np.random.choice(state.local_n_nets, sample_n_nets, replace=False)
                tmodels = [self.models[i] for i in idxs]

            t0 = time.time()
            losses = []
            steps = self.get_steps()

            # activate everything needed to run on this process
            grad_infos = []
            for model in tmodels:  # 多个随机模型进行梯度累计
                if state.train_nets_type == 'unknown_init':
                    # 如果是unknown_init则每一次都会reset
                    model.reset(state)

                l, saved = self.forward(model, rdata, rlabel, steps)
                losses.append(l)

                next_ones = self.backward(model, rdata, rlabel, steps, saved)
                grad_infos.append(next_ones)

            self.accumulate_grad(grad_infos)

            # all reduce if needed
            # average grad
            all_reduce_tensors = [p.grad for p in self.params]
            if do_log_this_iter:
                losses = torch.stack(losses, 0).sum()
                all_reduce_tensors.append(losses)

            if state.distributed:
                all_reduce_coalesced(all_reduce_tensors, grad_divisor)
            else:
                for t in all_reduce_tensors:
                    t.div_(grad_divisor)

            # opt step
            self.optimizer.step()
            t = time.time() - t0

            if do_log_this_iter:
                loss = losses.item()
                logging.info((
                    'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\t'
                    'Data Time: {:.2f}s\tTrain Time: {:.2f}s'
                ).format(
                    epoch, it * train_loader.batch_size, len(train_loader.dataset),
                    100. * it / len(train_loader), loss, data_t, t,
                ))
                if loss != loss:  # nan
                    raise RuntimeError('loss became NaN')
                    
            del steps, saved, grad_infos, losses, all_reduce_tensors

            data_t0 = time.time()

        with torch.no_grad():
            steps = self.get_steps()
        self.save_results(steps, visualize=state.visualize)
        return steps


def distill(state, models):
    return Trainer(state, models).train()
