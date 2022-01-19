import shutil
import time

import jittor as jt
import os
from tqdm import tqdm
import numpy as np


def train(model, optimizer, dataset, num_epochs, loss_fn, print_every, lr_schedule, model_dir):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    best_loss = float('inf')
    total_steps = 0
    model.train()
    with tqdm(total=num_epochs) as pbar:
        train_loss = []
        for epoch in range(num_epochs):

            if lr_schedule is not None:
                if (epoch + 1) % lr_schedule == 0:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.1

            start_time = time.time()

            model_input = {'coords': dataset['coords']}
            gt = dataset

            model_output = model(model_input)
            loss = loss_fn(model_output, gt)

            train_loss.append(loss.item())


            optimizer.step(loss)

            if loss.item() < best_loss:
                best_loss = loss.item()
                jt.save(model.state_dict(), os.path.join(model_dir, "model_final.pkl"))

            if (epoch + 1) % print_every == 0:
                tqdm.write("Epoch %d, loss %0.6f, best %0.6f, iteration time cost %0.6f" % (
                    epoch + 1, loss.item(), best_loss, time.time() - start_time))

            total_steps += 1
            pbar.update(1)

        np.savetxt(os.path.join(model_dir, "train_loss.txt"), np.array(train_loss))
