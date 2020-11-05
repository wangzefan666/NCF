import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter
import parse
import model
import config
import evaluate
import load_data

args = parse.args
# PREPARE DATASET #
train_data, test_data, user_num, item_num, train_mat = load_data.load_all()

# construct the train and test datasets
train_dataset = load_data.NCFData(train_data, item_num, train_mat, args.num_ng, True)
test_dataset = load_data.NCFData(test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset,  batch_size=args.test_num_ng + 1, shuffle=False)  # a batch is a testing user

# CREATE MODEL #
if config.model == 'NeuMF-pre':  # load pretrained model
    assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
    assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(config.GMF_model_path)
    MLP_model = torch.load(config.MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
                  args.dropout, config.model, GMF_model, MLP_model).cuda()
loss_function = nn.BCEWithLogitsLoss()  # sigmoid + BCELoss

if config.model == 'NeuMF-pre':  # fine tune pretrained model
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

writer = SummaryWriter()  # for visualization: loss

# TRAINING #
count, best_hr, best_epoch, best_ndcg = 0, 0, 0, 0
for epoch in range(args.epochs):
    model.train()  # Enable dropout (if have).
    start_time = time.time()
    # re sample ng_items for each user for training
    # class DataLoader will shuffle data at the beginning of each epoch
    train_loader.dataset.ng_sample()

    for user, item, label in train_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()
        model.zero_grad()
        prediction = model(user, item)
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
        count += 1
        writer.add_scalar('data/loss', loss.item(), count)

    with torch.no_grad():
        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
        elapsed_time = time.time() - start_time
        print("time of epoch {:03d}".format(epoch) + " is: ", elapsed_time)
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if args.save:
            if not os.path.exists(config.model_path):
                os.mkdir(config.model_path)
            torch.save(model, f'{config.model_path}{config.model}.pth')

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch, best_hr, best_ndcg))
