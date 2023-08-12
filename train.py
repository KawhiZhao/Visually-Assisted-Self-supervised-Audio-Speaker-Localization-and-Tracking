import torch
from torch import nn
from AV16_dataset.AV16 import *
from torch.utils.data import DataLoader
from models.audio_network import *
from utils import EarlyStopping
import numpy as np
import argparse

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(dataloader, model, loss_fn_1, loss_fn_2, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (mel, img, seg, gt, seq_name) in enumerate(dataloader):
    
        mel = mel.unsqueeze(1)
       
        
        mel = mel.to(device, dtype=float)
        mel = mel.type(torch.cuda.FloatTensor)
        gt = gt.to(device, dtype=torch.float32)
        seg = seg.to(device, dtype=torch.long)
        loc, rec = model(mel, seq_name, device)
        rec = rec.permute(0, 1, 3, 2)
        # rec = rec.reshape(64, 288 * 360, 2)
        # seg = seg.reshape(64, 288 * 360)
        loss1 = loss_fn_1(loc, gt)
        loss2 = loss_fn_2(rec, seg)
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss1, loss2, loss, current = loss1.item(), loss2.item(), loss.item(), batch * len(mel)
            print('loss: %f location loss: %f recons loss %f current %d / %d' % (loss, loss1, loss2, current, size))

def test(dataloader, model, loss_fn_1, loss_fn_2, optimizer):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_loss_1, test_loss_2 = 0, 0, 0
    dx_sum, dy_sum = 0, 0
    with torch.no_grad():
        for batch, (mel, img, seg, gt, seq_name) in enumerate(dataloader):
          
            mel = mel.unsqueeze(1)
            mel = mel.to(device, dtype=float)
            mel = mel.type(torch.cuda.FloatTensor)
            gt = gt.to(device, dtype=torch.float32)
            seg = seg.to(device, dtype=torch.long)
            loc, rec = model(mel, seq_name, device)
            rec = rec.permute(0, 1, 3, 2)
            loss1 = loss_fn_1(loc, gt)
            loss2 = loss_fn_2(rec, seg)
            test_loss_1 += loss1
            test_loss_2 += loss2
            test_loss += loss1 + loss2 
            dx = loc[:, 0] - gt[:, 0]
            dx = np.array(dx.cpu())
            dx = np.mean(np.abs(dx))
            dx_sum += dx
            dy = loc[:, 1] - gt[:, 1]
            dy = np.array(dy.cpu())
            dy = np.mean(np.abs(dy))
            dy_sum += dy
    test_loss /= num_batches
    test_loss_1 /= num_batches
    test_loss_2 /= num_batches
    dx_sum /= num_batches
    dy_sum /= num_batches
    print('Avg test loss for each batch: %f' % (test_loss))
    print('Avg localization loss for each batch: %f' % (test_loss_1))
    print('Avg reconstruction loss for each batch: %f' % (test_loss_2))    
    print('Avg dx for each batch: %f' % (dx_sum))
    print('Avg dy for each batch: %f' % (dy_sum))
    return test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default='5e-3',
                        help='which dataset')

    args = parser.parse_args()
    lr = float(args.lr)
    print('Loading training data....')
    training_data = AV16_dataset_gccphat_seg(data_split='train')
    
    training_dataloader = DataLoader(training_data, batch_size=1024, shuffle=True, num_workers=4)

    eval_data = AV16_dataset_gccphat_seg(data_split='eval')
    test_data = AV16_dataset_gccphat_seg(data_split='test')
    # print('Loading test data')
    eval_dataloader = DataLoader(eval_data, batch_size=1024, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=1024, shuffle=False, num_workers=4)

    model = audioNetwork_multi_task().to(device=device)
  
    loss_fn_1 = nn.MSELoss(reduction='mean')
    ce_weight = torch.tensor([1.0, 10.0])
    ce_weight = ce_weight.to(device=device)
    loss_fn_2 = nn.CrossEntropyLoss(weight=ce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoches = 5
    
    es = EarlyStopping(patience=30, verbose=True)

    for t in range(epoches):
        print('Epoch %d.....' % (t + 1))
        train(training_dataloader, model, loss_fn_1, loss_fn_2, optimizer)
        test_loss = test(eval_dataloader, model, loss_fn_1, loss_fn_2, optimizer)
        es(test_loss, model)
        if es.early_stop:
            print("Early stop...")
            break
    print('finish training')
