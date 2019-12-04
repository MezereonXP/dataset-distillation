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
from basics import task_loss, final_objective_loss, evaluate_steps
from contextlib import contextmanager
from utils.baselines import average_train
from skimage.measure import compare_ssim, compare_mse, compare_nrmse


def permute_list(list):
    indices = np.random.permutation(len(list))
    return [list[i] for i in indices]
def rvs(dim):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H
 
def distillation_label_distance_based_initialiser(state, distance_matrix):
    num_classes=state.num_classes
    if state.num_classes==2:
        dl_array = [[i==j for i in range(1)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
    else:
        dl_array = [[i==j for i in range(num_classes)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
    new_array = np.array(dl_array, dtype=float)
    
    #move the vectors closer based on distances
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if i!=j:
                new_array[i]=np.add(new_array[i],np.multiply(dl_array[j],(1-distance_matrix[i][j])))
    return new_array
    
        
def images_dist(dist_metric, images):
   imgs = np.moveaxis(np.array(images), 1, -1)
   dist_mat = np.zeros((len(imgs),len(imgs)))
   for i in range(len(imgs)):
       for j in range(len(imgs)):
           if dist_metric=="MSE":
               dist_mat[i,j] = compare_mse(imgs[i], imgs[j])
           elif dist_metric=="NRMSE":
               dist_mat[i,j] = compare_nrmse(imgs[i], imgs[j])
           elif dist_metric=="SSIM":
               dist_mat[i,j] = 1-compare_ssim(imgs[i], imgs[j], win_size=3, multichannel=True)
   return np.divide(dist_mat, np.max(dist_mat))
def distillation_label_initialiser(state, num_per_step, dtype, req_lbl_grad):
    init_type=state.random_init_labels
    device=state.device
    num_classes=state.num_classes
    label_smoothing=0.1
    if init_type=="stdnormal":
        if state.num_classes == 2:
            dl_array = np.random.normal(size=(num_per_step, 1))
            #dl_array = torch.randn(num_per_step, 1, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
        else:
            dl_array = np.random.normal(size=(num_per_step, num_classes))
            #dl_array = torch.randn(num_per_step, num_classes, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
    elif init_type=="uniform":
        if num_classes == 2:
            dl_array = np.random.uniform(size=(num_per_step, 1))
            #dl_array = torch.rand(num_per_step, 1, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
        else:
            dl_array = np.random.uniform(size=(num_per_step, num_classes))
            #dl_array = torch.rand(num_per_step, num_classes, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
    elif init_type=="zeros":
        if num_classes == 2:
            dl_array = np.zeros((num_per_step, 1))
        else:
            dl_array = np.zeros((num_per_step, num_classes))
    elif init_type=="ones":
        if num_classes == 2:
            dl_array = np.ones((num_per_step, 1))
            #distill_label = torch.ones(num_per_step, 1, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
        else:
            dl_array = np.ones((num_per_step, num_classes))
            #distill_label = torch.ones(num_per_step, num_classes, dtype=torch.float, device=device, requires_grad=req_lbl_grad)
    elif init_type=="hard":
        if state.num_classes==2:
            dl_array = [[i==j for i in range(1)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
        else:
            dl_array = [[i==j for i in range(num_classes)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
        #distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=device)
    elif init_type=="smoothed":
        if state.num_classes==2:
            dl_array = [[i==j for i in range(1)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
        else:
            dl_array = [[i==j for i in range(num_classes)]for j in range(num_classes)]*state.distilled_images_per_class_per_step
        dl_array=np.add(np.multiply(dl_array,(1-label_smoothing)), label_smoothing/num_classes)
        #distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=device)
    elif init_type=="orthogonal":
        M = rvs(num_classes)
        #This means that if you have multiple images per class per step, all labels for same class are same
        dl_array = M*state.distilled_images_per_class_per_step 
        #distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=device)
    elif init_type=="file":
        with open("labels.txt") as f:
            dl_array = [[float(l) for l in line.strip().split(", ")] for line in f.readlines()]
        #distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=device)
    elif init_type=="CNDB":
        distances=...
        dl_array=distillation_label_distance_based_initialiser(distances)
    elif init_type=="AIDB":
        avg_imgs = average_train(state)[0][0]
        distances= images_dist(state.dist_metric, avg_imgs)
        dl_array=distillation_label_distance_based_initialiser(state,distances)
    
    if state.add_first:
        dl_array=np.add(dl_array,state.add_label_scaling)
        dl_array=np.multiply(dl_array,state.mult_label_scaling)
    else:
        dl_array=np.multiply(dl_array,state.mult_label_scaling)
        dl_array=np.add(dl_array,state.add_label_scaling)
    distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=device)
    return distill_label

class Trainer(object):
    def __init__(self, state, models):
        self.state = state
        self.models = models
        self.num_data_steps = state.distill_steps  # how much data we have
        self.T = state.distill_steps * state.distill_epochs  # how many sc steps we run
        self.num_per_step = state.num_distill_classes * state.distilled_images_per_class_per_step
        assert state.distill_lr >= 0, 'distill_lr must >= 0'
        assert len(state.init_labels)==state.num_distill_classes, 'len(init_labels) must == num_distill_classes'
        self.init_data_optim()

    def init_data_optim(self):
        self.params = []
        state = self.state
        optim_lr = state.lr
        req_lbl_grad = not state.static_labels
        # labels
        self.labels = []
        
        #distill_label = distill_label.t().reshape(-1)  # [0, 0, ..., 1, 1, ...]
        #distill_label = torch.nn.Softmax(distill_label, dim=1)
        for _ in range(self.num_data_steps):
            if state.random_init_labels:
                distill_label = distillation_label_initialiser(state, self.num_per_step, torch.float, req_lbl_grad)
            else:
                if state.num_classes==2:
                    dl_array = [[i==j for i in range(1)]for j in state.init_labels]*state.distilled_images_per_class_per_step
                else:
                    dl_array = [[i==j for i in range(state.num_classes)]for j in state.init_labels]*state.distilled_images_per_class_per_step
                
                
                distill_label=torch.tensor(dl_array,dtype=torch.float, requires_grad=req_lbl_grad, device=state.device)
                    
                #distill_label = self.one_hot_embedding(distill_label, state.num_classes)
                             
            if not state.static_labels:
                self.labels.append(distill_label)
                self.params.append(distill_label)
            else:
                self.labels.append(distill_label)
        self.all_labels = torch.cat(self.labels)

        # data
        self.data = []
        for _ in range(self.num_data_steps):
            if state.textdata:
                distill_data = torch.randn(self.num_per_step, state.nc, state.input_size, state.ninp,
                                       device=state.device, requires_grad=True)
            else:
                distill_data = torch.randn(self.num_per_step, state.nc, state.input_size, state.input_size,
                                       device=state.device, requires_grad=True)
            self.data.append(distill_data)
            self.params.append(distill_data)

        # lr

        # undo the softplus + threshold
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
    def one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.
    
        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.
    
        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        req_lbl_grad = not self.state.static_labels
        emb = torch.nn.Embedding(num_classes, num_classes)
        emb.weight.data = torch.eye(num_classes)
        emb.weight.requires_grad=True
        emb=emb.to(self.state.device)
        return emb(labels)
    def get_steps(self):
        data_label_iterable = (x for _ in range(self.state.distill_epochs) for x in zip(self.data, self.labels))
        lrs = F.softplus(self.raw_distill_lrs).unbind()
        steps = []
        for (data, label), lr in zip(data_label_iterable, lrs):
            steps.append((data,label, lr))
        return steps

    def forward(self, model, rdata, rlabel, steps):
        state = self.state

        # forward
        model.train()
        w = model.get_param()
        params = [w]
        gws = []
        for step_i, (data, label, lr) in enumerate(steps):
            with torch.enable_grad():
                model.distilling_flag=True
                output = model.forward_with_param(data, w)
                loss = task_loss(state, output, label)
            gw, = torch.autograd.grad(loss, w, lr, create_graph=True)

            with torch.no_grad():
                new_w = w.sub(gw).requires_grad_()
                params.append(new_w)
                gws.append(gw)
                w = new_w

        # final L
        model.eval()
        model.distilling_flag=False
        output = model.forward_with_param(rdata, params[-1])
        ll = final_objective_loss(state, output, rlabel)
        return ll, (ll, params, gws)

    def backward(self, model, rdata, rlabel, steps, saved_for_backward):
        l, params, gws = saved_for_backward
        state = self.state

        datas = []
        gdatas = []
        lrs = []
        glrs = []
        labels=[]
        glabels=[]
        dw, = torch.autograd.grad(l, (params[-1],), retain_graph=state.textdata)

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
        for (data, label, lr), w, gw in reversed(list(zip(steps, params, gws))):
            # hvp_in are the tensors we need gradients w.r.t. final L:
            #   lr (if learning)
            #   data
            #   ws (PRE-GD) (needed for next step)
            #
            # source of gradients can be from:
            #   gw, the gradient in this step, whose gradients come from:
            #     the POST-GD updated ws
            hvp_in = [w]
            hvp_in.append(data)
            hvp_in.append(lr)
            if not state.static_labels:
                hvp_in.append(label)
            dgw = dw.neg()  # gw is already weighted by lr, so simple negation
            hvp_grad = torch.autograd.grad(
                outputs=(gw,),
                inputs=hvp_in,
                grad_outputs=(dgw,),
                retain_graph=state.textdata
            )
            # Update for next iteration, i.e., previous step
            with torch.no_grad():
                # Save the computed gdata and glrs
                datas.append(data)
                gdatas.append(hvp_grad[1])
                lrs.append(lr)
                glrs.append(hvp_grad[2])
                if not state.static_labels:
                    labels.append(label)
                    glabels.append(hvp_grad[3])

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
            for d, g in zip(datas, gdatas):
                d.grad.add_(g)
            if not self.state.static_labels:
                for l, g in zip(labels, glabels):
                    l.grad.add_(g)
        if len(bwd_out) > 0:
            torch.autograd.backward(bwd_out, bwd_grad)#MULTISTEP PROBLEM?

    def save_results(self, steps=None, visualize=True, subfolder=''):
        with torch.no_grad():
            steps = steps or self.get_steps()
            save_results(self.state, steps, visualize=visualize, subfolder=subfolder)

    def __call__(self):
        return self.train()

    def prefetch_train_loader_iter(self):
        state = self.state
        device = state.device
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
                    #print(example.fields)
                    data = example.text[0]
                    target = example.label
                    val=(data,target)
                else:
                    val=example
                
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
            for model in tmodels:
                if state.train_nets_type == 'unknown_init':
                    model.reset(state)
                l, saved = self.forward(model, rdata, rlabel, steps)
                losses.append(l)
                grad_infos.append(self.backward(model, rdata, rlabel, steps, saved))
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
