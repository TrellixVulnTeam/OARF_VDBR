import numpy as np
from scipy.optimize import newton
import argparse
import time
import copy
import math
import sys
import os
import random
import importlib

parser = argparse.ArgumentParser(description='Handwritten Chinese Character Recognition')
parser.add_argument('--task', default='chinese', help="ML Task")
parser.add_argument('--setting', default='fedavg', help="Training setting (0|1|...|combined|fedavg), where the number represents the party")
parser.add_argument('--dp', action='store_true', default=False, help='Enable DP')
parser.add_argument('--spdz', action='store_true', default=False, help='Enable SPDZ')
parser.add_argument('--pysyft-remote-training', action='store_true', default=False, help='Enable PySyft remote training (buggy for now)')
parser.add_argument('-e', '--epsilon', default=1.0, type=float, help="Privacy Budget for each party")  # experiment variable
parser.add_argument('--lotsize-scaler', default=1.0, type=float, help="Scale the lot size sqrt(N) by a multiplier")  # experiment variable
parser.add_argument('-c', '--clip', default=1.0, type=float, help="L2 bound for the gradient clip")  # experiment variable
parser.add_argument('-E', '--epochs', default=10, type=int, help="Number of epochs")
parser.add_argument('--local-epochs', default=1, type=int, help="FedAvg per how many epochs")
parser.add_argument('-b', '--batch-size', default=128, type=int, help="How many records per batch")
parser.add_argument('--val-batch-size', default=256, type=int, help="Validation and testing set batch size")
parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
parser.add_argument('--patience', default=5, type=float, help="Patience for early stopping")
parser.add_argument('--min_delta', default=0, type=float, help="Min delta for early stopping")
parser.add_argument('--test-freq', default=5, type=int, help="Test per how many epochs")

parser.add_argument('--gpu', action='store_true', default=True, help="Use gpu")
parser.add_argument('--which-gpu', default="0", help="Use which gpu")
parser.add_argument('--seed', default=0, type=int, help="Random seed")
parser.add_argument('--load-model', default=None, help="Load trained model, e.g. chinese/models/epo19.pt")
parser.add_argument('--starting-epoch', default=0, type=int, help="Start from the beginning of which epoch, e.g. 20")

args = parser.parse_args()

assert args.setting in ['combined', 'fedavg'] or args.setting.isdigit(), 'Setting not supported'
if args.spdz: assert args.setting == 'fedavg'
assert args.pysyft_remote_training == False, 'PySyft remote training not supported for now'

os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpu

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import h5py as h5
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import RandomSampler, DataLoader, Dataset
from torch.autograd import Variable
import syft as sy

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
set_seed(args.seed)

# import task modules
dataset_path = '%s/dataset.py' % (args.task)
model_path = '%s/model.py' % (args.task)
assert os.path.exists(dataset_path) and os.path.exists(model_path), 'Please place dataset.py and model.py under the task directory'
dataset_module = importlib.import_module('%s.dataset' % (args.task))
model_module = importlib.import_module('%s.model' % (args.task))
get_loaders = getattr(dataset_module, 'get_loaders')
get_model = getattr(model_module, 'get_model')
get_loss_func = getattr(model_module, 'get_loss_func')  # e.g. log_loss
get_metric_func = getattr(model_module, 'get_metric_func')  # e.g. accuracy

loss_func = get_loss_func(args)
metric_func = get_metric_func(args)

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Task:', args.task)
print('Setting:', args.setting)
if args.spdz: print('Using SPDZ for FedAvg')
print('Local epochs:', args.local_epochs)

all_trn_loaders, val_loaders, tst_loaders = get_loaders(args)  # loaders contain each party's loader and the combined loader
trn_party_loaders, trn_combined_loader = all_trn_loaders[:-1], all_trn_loaders[-1]
val_party_loaders, val_combined_loader = val_loaders[:-1], val_loaders[-1]
tst_party_loaders, tst_combined_loader = tst_loaders[:-1], tst_loaders[-1]

if args.setting == 'fedavg':
    trn_loaders = trn_party_loaders
    val_loaders = val_party_loaders
    tst_loaders = tst_party_loaders
else:
    if args.setting == 'combined':
        party = -1
    else:  # single party
        party = int(args.setting)
    trn_loaders = [trn_loaders[party]]
    val_loaders = [val_loaders[party]]
    tst_loaders = [tst_loaders[party]]

lengths = [len(loader.dataset) for loader in trn_loaders]  # list of party's dataset length
num_parties = len(lengths)
total_length = sum(lengths)

def avg(*numbers):  # weighted average to compute weighted metric or loss
    assert len(numbers) == len(lengths), "Input length must be the number of parties"
    return sum([n * l for n, l in zip(numbers, lengths)]) / total_length

precision = 1000
ratios = [round(precision * length / total_length) for length in lengths]  # integer share used by SPDZ

batches_per_lot_list = [None] * num_parties
sigma_list = [None] * num_parties
if args.dp:
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    def find_sigma(eps, batches_per_lot, dataset_size):
        lotSize = batches_per_lot * args.batch_size # L
        N = dataset_size
        delta = min(10**(-5), 1 / N)
        lotsPerEpoch = N / lotSize
        q = lotSize / N  # Sampling ratio
        T = args.epochs * lotsPerEpoch  # Total number of lots

        def compute_dp_sgd_wrapper(_sigma):
            with HiddenPrints():
                return compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=N, batch_size=lotSize, noise_multiplier=_sigma, epochs=args.epochs, delta=delta)[0] - args.epsilon

        sigma = newton(compute_dp_sgd_wrapper, x0=0.5, tol=1e-4)  # adjust x0 to avoid error
        with HiddenPrints():
            actual_eps = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=N, batch_size=lotSize, noise_multiplier=sigma, epochs=args.epochs, delta=delta)[0]
#         print('Batches_per_lot={}, q={}, T={}, sigma={}'.format(batches_per_lot, q, T, sigma))
#         print('actual epslion = {}'.format(actual_eps))
        return sigma

    print('Epsilon:', args.epsilon)
    lotsizes = [len(loader.dataset)**.5 * args.lotsize_scaler for loader in trn_loaders]
    batches_per_lot_list = list(map(lambda lotsize: max(round(lotsize / args.batch_size), 1), lotsizes))
    batches_per_lot_list = [min(bpl, len(loader)) for bpl, loader in zip(batches_per_lot_list, trn_loaders)]
    print('Batches per lot:', batches_per_lot_list)
    sigma_list = [find_sigma(args.epsilon, bpl, len(loader.dataset)) for bpl, loader in zip(batches_per_lot_list, trn_loaders)]
    print('Sigma:', sigma_list)

_lastNoiseShape = None
_noiseToAdd = None

def divide_clip_grads(model, batch_per_lot=None):
    assert args.dp == True
    for key, param in model.named_parameters():
        if batch_per_lot is None:
            param.grad /= model.batch_per_lot
        else:
            param.grad /= batch_per_lot
        nn.utils.clip_grad_norm([param], args.clip)

def gaussian_noise(model, grads):
    global _lastNoiseShape
    global _noiseToAdd
    if grads.shape != _lastNoiseShape:
        _lastNoiseShape = grads.shape
        _noiseToAdd = torch.zeros(grads.shape).to(device)
    _noiseToAdd.data.normal_(0.0, std=args.clip*model.sigma)
    return _noiseToAdd

def add_noise_to_grads(model, batch_per_lot=None):
    assert args.dp == True
    for key, param in model.named_parameters():
        if batch_per_lot is None:
            lotsize = model.batch_per_lot * args.batch_size
        else:
            lotsize = batch_per_lot * args.batch_size
        noise = 1/lotsize * gaussian_noise(model, param.grad)
        param.grad += noise

model = get_model(args)

if args.load_model is not None:
    model.load_state_dict(torch.load(args.load_model))
    print('Loaded model:', args.load_model)

models = [model]
models += [copy.deepcopy(model) for _ in range(num_parties - 1)]

if args.dp:
    assert len(models) == num_parties  # debug
    assert len(batches_per_lot_list) == num_parties
    assert len(sigma_list) == num_parties
    for mod, bpl, sig in zip(models, batches_per_lot_list, sigma_list):
        mod.batch_per_lot = bpl
        mod.sigma = sig

optims = []
for i in range(len(models)):
    models[i] = models[i].to(device)
    optims.append(optim.Adam(models[i].parameters(), lr=args.lr))
    params = [list(mod.parameters()) for mod in models]  # used by fedavg

if args.setting == 'fedavg':
    # use PySyft for SPDZ
    hook = sy.TorchHook(torch)
    party_workers = [sy.VirtualWorker(hook, id="party{:d}".format(i)) for i in range(num_parties)]
    crypto = sy.VirtualWorker(hook, id="crypto")

def train(epoch):
    for mod in models:
        mod.train()
    
    losses = [0] * num_parties
    metrics = [0] * num_parties
    
    def trn_batch(data, target, model, optimizer, party_i, batch_i, batch_per_lot):
        data, target = data.to(device), target.to(device)
        
        if args.dp:
            if batch_i % batch_per_lot == 0:
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
        
        output = model(data)
        loss = loss_func(output, target)
        losses[party_i] += loss_func(output, target, reduction='sum').item()
        metrics[party_i] += metric_func(output, target)
        loss.backward()
        
        if args.dp:
            if batch_i % batch_per_lot == batch_per_lot - 1:
                divide_clip_grads(model)
                add_noise_to_grads(model)
                optimizer.step()
            elif (batch_i == len(trn_loaders[party_i]) - 1):  # reach the end of the last incomplete lot
                divide_clip_grads(model, batch_i % batch_per_lot + 1)
                add_noise_to_grads(model, batch_i % batch_per_lot + 1)
                optimizer.step()
        else:
            optimizer.step()
    
    for party_i, (loader, mod, optim, bpl) in enumerate(zip(trn_loaders, models, optims, batches_per_lot_list)):
        for batch_i, (data, target) in enumerate(loader):
            trn_batch(data, target, mod, optim, party_i, batch_i, bpl)
    
    loss_print = 'Trn loss: '
    metric_print = 'Trn metric: '
    for i in range(num_parties):
        losses[i] /= len(trn_loaders[i].dataset)
        metrics[i] /= len(trn_loaders[i].dataset)
        loss_print += '{:4f} '.format(losses[i])
        metric_print += '{:4f} '.format(metrics[i])
    print(loss_print)
    print(metric_print)

    if args.setting == 'fedavg' and (epoch % args.local_epochs == args.local_epochs - 1 or epoch == args.epochs - 1):
        if args.local_epochs > 1:
            print('Fedavg now')
        if args.spdz:
            new_params = list()
            for param_i in range(len(params[0])):
                spdz_params = list()
                for party_i in range(num_parties):
                    spdz_params.append(params[party_i][param_i].copy().cpu().fix_precision().share(*party_workers, crypto_provider=crypto))
                new_param = sum([p * r for p, r in zip(spdz_params, ratios)]).get().float_precision() / sum(ratios)
                new_params.append(new_param)

            with torch.no_grad():
                for model_params in params:
                    for param in model_params:
                        param *= 0

                for param_index in range(len(params[0])):
                    if str(device) == 'cpu':
                        for model_params in params:
                            model_params[param_index].set_(new_params[param_index])
                    else:
                        for model_params in params:
                            model_params[param_index].set_(new_params[param_index].cuda())
        else:
            with torch.no_grad():
                for ps in zip(*params):
                    p_avg = sum([p.data * r for p, r in zip(ps, ratios)]) / sum(ratios)
                    for p in ps:
                        p.set_(p_avg)

def val(loaders):
    global min_loss
    global wait
    
#     model.eval()  # doesn't work right
    losses = [0] * num_parties
    metrics = [0] * num_parties
    
    def val_batch(data, target, model, party_i):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)    # get the index of the max log-probability
        metrics[party_i] += pred.eq(target.view_as(pred)).sum().item()
        losses[party_i] += loss_func(output, target, reduction='sum').item()
    
    for party_i, (loader, mod) in enumerate(zip(loaders, models)):
        for data, target in loader:
            val_batch(data, target, mod, party_i)

    loss_print = 'Eval loss: '
    metric_print = 'Eval metric: '
    for i in range(num_parties):
        losses[i] /= len(trn_loaders[i].dataset)
        metrics[i] /= len(trn_loaders[i].dataset)
        loss_print += '{:4f} '.format(losses[i])
        metric_print += '{:4f} '.format(metrics[i])
    print(loss_print)
    print(metric_print)
    
    loss_avg = avg(*losses)
    metric_avg = avg(*metrics)
    if num_parties > 1:
        print('Eval loss_avg {:.4f}, metric_avg {:.4f}'.format(loss_avg, metric_avg))
    
    if min_loss - loss_avg > args.min_delta:
        min_loss = loss_avg
        wait = 0
    else:
        wait += 1

# save model to
model_dir = '{}/models/setting-{}-epochs-{}-localepochs-{}'.format(args.task, args.setting, args.epochs, args.local_epochs)
if args.dp:
    model_dir += '-eps-{}-lotsize_scaler-{}'.format(args.epsilon, args.lotsize_scaler)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
# Train
min_loss = float('inf')
wait = 0
for epoch in range(args.starting_epoch, args.epochs):
    print('Epoch', epoch)
    t1 = int(time.time())
    train(epoch)
    t2 = int(time.time())
    val(val_loaders)
    if args.test_freq > 0 and epoch % args.test_freq == args.test_freq - 1:
        print('On test set')
        val(tst_loaders)
        torch.save(models[0].state_dict(), model_dir + "/epoch-{}.pt".format(epoch))
        print('Saved model to:', model_dir + "/epoch-{}.pt".format(epoch))
    t3 = int(time.time())
    print('Epoch trn time {:d}s, val time {:d}s'.format(t2-t1, t3-t2))
    if wait == args.patience:
        print('Early stop')
        break

torch.save(models[0].state_dict(), model_dir + "/epoch-{}.pt".format(args.epochs - 1))
