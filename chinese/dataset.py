import h5py as h5
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import RandomSampler, DataLoader, Dataset

def get_loaders(args):
    # load data
    class HCC(Dataset):  # Handwritten Chinese character dataset
        # group: trn/vld
        def __init__(self, archive, group, transform=None):
            self.trn = True if group == 'trn' else False
            self.archive = h5.File(archive, 'r')
            self.x = self.archive[group + '/x']
            self.y = self.archive[group + '/y']
            self.transform = transform
        def __getitem__(self, index):
            datum = self.x[index]
            if self.transform is not None:
                datum = self.transform(datum)
            label = self.y[index][0].astype('int64')
            return datum, label
        def __len__(self):
            if self.trn and args.dp:
                return len(self.y) - len(self.y) % args.batch_size
            else:
                return len(self.y)
        def close(self):
            self.archive.close()
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.gpu else {}
        
    def to_tensor(img):
        img = torch.from_numpy(img)
        return img.float().div(255)
    tfm = transforms.Lambda(to_tensor)

    trnset_casia = HCC(args.task + '/data/HWDB1.1fullset.hdf5', 'trn', transform=transforms.Compose([tfm]))
    trnset_hit = HCC(args.task + '/data/HIT_OR3Cfullset.hdf5', 'trn', transform=transforms.Compose([tfm]))
    trnset_combined = HCC(args.task + '/data/HIT_HWDB1.1_fullset.hdf5', 'trn', transform=transforms.Compose([tfm]))
    if args.dp:
        sampler_casia = RandomSampler(trnset_casia, replacement=True)
        sampler_hit = RandomSampler(trnset_hit, replacement=True)
        sampler_combined = RandomSampler(trnset_combined, replacement=True)
        trn_loader_casia = DataLoader(trnset_casia, batch_size=args.batch_size, shuffle=False, sampler=sampler_casia, **kwargs)
        trn_loader_hit = DataLoader(trnset_hit, batch_size=args.batch_size, shuffle=False, sampler=sampler_hit, **kwargs)
        trn_loader_combined = DataLoader(trnset_combined, batch_size=args.batch_size, shuffle=False, sampler=sampler_hit, **kwargs)
    else:
        trn_loader_casia = DataLoader(trnset_casia, batch_size=args.batch_size, shuffle=True, **kwargs)
        trn_loader_hit = DataLoader(trnset_hit, batch_size=args.batch_size, shuffle=True, **kwargs)
        trn_loader_combined = DataLoader(trnset_combined, batch_size=args.batch_size, shuffle=True, **kwargs)

    valset_casia = HCC(args.task + '/data/HWDB1.1fullset.hdf5', 'vld', transform=transforms.Compose([tfm]))
    valset_hit = HCC(args.task + '/data/HIT_OR3Cfullset.hdf5', 'vld', transform=transforms.Compose([tfm]))
    valset_combined = HCC(args.task + '/data/HIT_HWDB1.1_fullset.hdf5', 'vld', transform=transforms.Compose([tfm]))
    val_loader_casia = DataLoader(valset_casia, batch_size=args.val_batch_size, shuffle=False, **kwargs)
    val_loader_hit = DataLoader(valset_hit, batch_size=args.val_batch_size, shuffle=False, **kwargs)
    val_loader_combined = DataLoader(valset_combined, batch_size=args.val_batch_size, shuffle=False, **kwargs)

    tstset_casia = HCC(args.task + '/data/HWDB1.1fullset.hdf5', 'tst', transform=transforms.Compose([tfm]))
    tstset_hit = HCC(args.task + '/data/HIT_OR3Cfullset.hdf5', 'tst', transform=transforms.Compose([tfm]))
    tstset_combined = HCC(args.task + '/data/HIT_HWDB1.1_fullset.hdf5', 'tst', transform=transforms.Compose([tfm]))
    tst_loader_casia = DataLoader(tstset_casia, batch_size=args.val_batch_size, shuffle=False, **kwargs)
    tst_loader_hit = DataLoader(tstset_hit, batch_size=args.val_batch_size, shuffle=False, **kwargs)
    tst_loader_combined = DataLoader(tstset_combined, batch_size=args.val_batch_size, shuffle=False, **kwargs)

    print('CASIA trn:', len(trnset_casia), len(trn_loader_casia))
    print('HIT trn:', len(trnset_hit), len(trn_loader_hit))
    print('COMBINED trn:', len(trnset_combined), len(trn_loader_combined))
    print('CASIA val:', len(valset_casia), len(val_loader_casia))
    print('HIT val:', len(valset_hit), len(val_loader_hit))
    print('COMBINED val:', len(valset_combined), len(val_loader_combined))
    print('CASIA tst:', len(tstset_casia), len(tst_loader_casia))
    print('HIT tst:', len(tstset_hit), len(tst_loader_hit))
    print('COMBINED tst:', len(tstset_combined), len(tst_loader_combined))
    
    trn_loaders = [trn_loader_casia, trn_loader_hit, trn_loader_combined]
    val_loaders = [val_loader_casia, val_loader_hit, val_loader_combined]
    tst_loaders = [tst_loader_casia, tst_loader_hit, tst_loader_combined]
    
    return trn_loaders, val_loaders, tst_loaders
