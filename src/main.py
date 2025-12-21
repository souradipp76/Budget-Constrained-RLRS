import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch import nn, optim

from dataset import input_generator
from model import DrrModel, LRSchedulerPerStep

# Configuration parameters (replacing TensorFlow FLAGS)
class Config:
    train = True
    train_set = "../data/rec_train_set.sample.txt"
    validation_set = "../data/rec_validation_set.sample.txt"
    test_set = "../data/rec_test_set.sample.txt"
    log_dir = "../log/"
    saved_model_name = "../outputs/drr_model.pth"
    model_type = 1
    batch_size = 32
    seq_len = 30
    train_epochs = 1
    train_steps_per_epoch = 100
    validation_steps = 10
    early_stop_patience = 10
    lr_per_step = 4000
    d_feature = 83
    d_model = 2
    d_inner_hid = 2
    n_head = 1
    d_k = 2
    d_v = 2
    n_layers = 1
    dropout = 0.1
    pos_embedding_mode = 1  # 0: fixed PE, 1: learnable PE, 2: no PE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training function
def train(CONFIG):
    print("Training...")
    os.makedirs(CONFIG.log_dir, exist_ok=True)
    device = CONFIG.device
    model = DrrModel(CONFIG.seq_len, CONFIG.d_feature, CONFIG.d_model, CONFIG.d_inner_hid, CONFIG.n_head, CONFIG.d_k, CONFIG.d_v, CONFIG.n_layers, CONFIG.dropout, CONFIG.model_type, CONFIG.pos_embedding_mode)
    if os.path.isfile(CONFIG.saved_model_name):
        print(f"Loading model from {CONFIG.saved_model_name}")
        model.load_state_dict(torch.load(CONFIG.saved_model_name, weights_only=True))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    running_loss = []

    # Initialize schedulers
    per_step_scheduler = LRSchedulerPerStep(optimizer, d_model=CONFIG.d_model, warmup=CONFIG.lr_per_step)

    # Training loop
    for epoch in range(CONFIG.train_epochs):
        model = model.to(device)
        model.train()
        running_loss_per_epoch = 0
        train_gen = input_generator(CONFIG.train_set, CONFIG.batch_size, CONFIG.seq_len, CONFIG.model_type)
        for step, (features, labels) in enumerate(train_gen):
            pos_input, v_input = features
            labels = labels.to(device)
            pos_input = pos_input.to(device)
            v_input = v_input.to(device)
            if step >= CONFIG.train_steps_per_epoch:
                break
            optimizer.zero_grad()
            outputs, _ = model(v_input, pos_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            per_step_scheduler.step()

            running_loss_per_epoch += loss.item()

            if (step+1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{CONFIG.train_epochs}, Step {step + 1}/{CONFIG.train_steps_per_epoch}, Loss: {running_loss_per_epoch / (step + 1)}")
        loss_per_epoch = running_loss_per_epoch / (step + 1)
        running_loss.append(loss_per_epoch)
        print(f"Epoch {epoch + 1}/{CONFIG.train_epochs} completed.")
        torch.save(model.state_dict(), CONFIG.saved_model_name)

        model.eval()
        val_gen = input_generator(CONFIG.validation_set, CONFIG.batch_size, CONFIG.seq_len, CONFIG.model_type)
        losses = []
        for step, (features, labels) in enumerate(val_gen):
            pos_input, v_input = features
            labels = labels.to(device)
            pos_input = pos_input.to(device)
            v_input = v_input.to(device)
            if step >= CONFIG.validation_steps:
                break
            outputs, _ = model(v_input, pos_input)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
        print(f"Validation Loss: {np.mean(losses)}")

    # Prediction function
def predict(CONFIG, mode = 'test'):
    print("Predicting...")
    model = DrrModel(CONFIG.seq_len, CONFIG.d_feature, CONFIG.d_model, CONFIG.d_inner_hid, CONFIG.n_head, CONFIG.d_k, CONFIG.d_v, CONFIG.n_layers, CONFIG.dropout, CONFIG.model_type, CONFIG.pos_embedding_mode)
    model.load_state_dict(torch.load(CONFIG.saved_model_name, weights_only=True))
    model.eval()

    if mode == 'test':
        gen = input_generator(CONFIG.test_set, CONFIG.batch_size, CONFIG.seq_len, CONFIG.model_type, shuffle = True)
    else:
        gen = input_generator(CONFIG.train_set, CONFIG.batch_size, CONFIG.seq_len, CONFIG.model_type, shuffle = True)
    preds = []
    for features, label_batch in gen:
        pos_enc, x = features
        predict_batch, _ = model(x, pos_enc)
        for labels, predicts in zip(label_batch, predict_batch):
            preds.append(predicts.detach().numpy())
            if sum(labels) > 0:  # predict valid labels
                new_ranks = np.argsort(-predicts.detach().numpy())
                new_labels = labels[new_ranks]
                with open(f"../outputs/{mode}_predict.out", "a") as f:
                    f.write("%s\t%s\n" % (json.dumps(labels.tolist()), json.dumps(new_labels.tolist())))
    return np.vstack(preds)

def calc_average_precision_at_k(labels, k):
    n = min(len(labels),k)
    labels = labels[:n]
    p = []
    p_cnt = 0
    for i in range(n):
        if labels[i]>0:
            p_cnt+=1
            p.append(p_cnt*1.0/(i+1))
    if p_cnt > 0:
        return sum(p)/p_cnt
    else:
        return 0.0

def calc_precision_at_k(labels, k):
    n = min(len(labels),k)
    labels = labels[:n]
    p_cnt = 0
    for i in range(n):
        if labels[i]>0:
            p_cnt+=1
    return p_cnt*1.0/n

def make_metric_dict():
    return { 'p@5':0, 'p@10':0, 'p@1':0, 'map@5':0, 'map@10':0, 'map@30':0 }

def calc_metrics():
    metric_keys = [ 'p@5', 'p@10', 'map@5', 'map@10', 'map@30' ]
    filename = "../outputs/train_predict.out"
    print("calc metric from %s" % filename)
    with open(filename, 'r') as f:
        cnt = 0
        d = {} # for stat
        for line in f:
            try:
                step_labels = [ json.loads(labels) for labels in line.strip().split("\t") ]
            except:
                print(line)
                continue
            n = len(step_labels)
            for i in range(n):
                if not i in d:
                    d[i] = make_metric_dict()
                d[i]['p@5'] += calc_precision_at_k(step_labels[i], 5)
                d[i]['p@10'] += calc_precision_at_k(step_labels[i], 10)
                d[i]['map@5'] += calc_average_precision_at_k(step_labels[i], 5)
                d[i]['map@10'] += calc_average_precision_at_k(step_labels[i], 10)
                d[i]['map@30'] += calc_average_precision_at_k(step_labels[i], 30)
            cnt+=1
        f.close()
        n = len(d)
        print('total_record_cnt=%d step_range=[0,%d]' % (cnt, n-1))
        for i in range(n):
            info = ["step=%d" % i]
            for key in metric_keys:
                info.append("%s=%0.2f" % (key, d[i][key] * 100.0/cnt))
            print(" ".join(info))

if __name__ == "__main__":
    CONFIG = Config()
    
    train_gen = input_generator(CONFIG.train_set, CONFIG.batch_size, CONFIG.seq_len, CONFIG.model_type)
    batch = next(train_gen)
    print(batch[0][0].shape)
    print(batch[0][1].shape)
    print(batch[1].shape)
    print(batch[0][1][0][0])
    
    if CONFIG.train:
        start_time = time.time()
        # train(CONFIG)
        print(f"Job done! Time taken: {round((time.time() - start_time) / 60, 2)} minutes.")
    start_time = time.time()
    CONFIG.train = False
    train_preds = predict(CONFIG, 'train')
    print(train_preds.shape)
    print(f"Job done! Time taken: {round((time.time() - start_time) / 60, 2)} minutes.")

    calc_metrics()
    