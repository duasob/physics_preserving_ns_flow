# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
utils
"""
import time
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

import torch


def _calculate_error(label, prediction, batch_size):
    """calculate l2-error to evaluate accuracy"""
    rel_error = np.sqrt(np.sum(np.square(label.reshape(batch_size, -1) - prediction.reshape(batch_size, -1)))) / \
                np.sqrt(np.sum(np.square(label.reshape(batch_size, -1))))
    return rel_error


def calculate_l2_error(model, inputs, label, batch_size, device="cuda"):
    """
    Evaluate the model respect to input data and label.

    Args:
        model (Cell): Prediction network cell.
        inputs (Array): Input data of prediction.
        label (Array): Label data of prediction.
        batch_size (int): size of prediction batch.
    """
    print("================================Start Evaluation================================")
    time_beg = time.time()
    rel_rmse_error = 0.0
    prediction = 0.0
    length = label.shape[0]
    t = 10
    for i in range(length):
        for j in range(t - 1, t + 9):
            cur_label = label[i:i + 1, j]
            if j == t - 1:
                test_batch = torch.as_tensor(inputs[i:i + 1, j], dtype=torch.float32, device=device)
            else:
                test_batch = torch.as_tensor(np.expand_dims(prediction, axis=-2), dtype=torch.float32, device=device)
            with torch.no_grad():
                prediction = model(test_batch[..., -1, :]).detach().cpu().numpy()
            rel_rmse_error_step = _calculate_error(cur_label[..., -1, :], prediction, batch_size)
            rel_rmse_error += rel_rmse_error_step

    rel_rmse_error = rel_rmse_error / (length * 10)
    print("mean rel_rmse_error:", rel_rmse_error)
    print("=================================End Evaluation=================================")
    print(f"predict total time: {time.time() - time_beg} s")
    return rel_rmse_error

def visual_animate(yy, yp, ye):
    """ Plot animate figures.

    Args:
        yy (numpy.array): Label data with shape e.g. :math:`[T, C, H, W]`.
        yp (numpy.array): Label data with shape e.g. :math:`[T, C, H, W]`.
        ye (numpy.array): Error data with shape e.g. :math:`[T, C, H, W]`.
    """
    cmap = matplotlib.colormaps['jet']
    fig, ax = plt.subplots(1, 3, figsize=[7, 3])

    ax[0].set_title('Label')
    im0 = ax[0].imshow(yy[0], cmap=cmap)
    ax[1].set_title('Prediction')
    im1 = ax[1].imshow(yp[0], cmap=cmap)
    ax[2].set_title('Error')
    im2 = ax[2].imshow(ye[0], cmap=cmap)
    title = fig.suptitle('t=0')
    fig.tight_layout()
    fig.colorbar(im1, ax=ax)

    def animate(i):
        # check that the i is within bounds
        if i < len(yy) and i < len(yp) and i < len(ye):
            y, p, e = yy[i], yp[i], ye[i]
        else:
            return

        im0.set_data(y)
        im1.set_data(p)
        im2.set_data(e)
        vmin, vmax = np.min(y), np.max(y)
        im0.set_clim(vmin, vmax)
        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)
        title.set_text(f't={i}')

    ani = animation.FuncAnimation(fig, animate, interval=200, blit=False, frames=10,
                                  repeat_delay=1000)
    ani.save('images/result.gif', writer='imagemagick')
    plt.close(fig)

def visual(model, inputs, labels_unif, data_config, device="cuda"):
    """ Infer the model sequentially and visualize the results.

    Args:
        Prediction network cell.
        inputs (numpy.array): Input data, interpolated on Gauss grid.
        labels_unif (numpy.Array): original label data.
        data_config (dict): dict with data configurations.
    """
    if not os.path.exists('images'):
        os.makedirs('images')

    res = data_config['resolution']

    prediction = None

    t = 10  # number of time steps to predict sequentially
    pred_unif = []
    for j in range(t - 1, t + 9):
        unif_label = labels_unif[0, j]
        if j == t - 1:
            test_batch = torch.as_tensor(
                inputs[0, j].reshape((1, res, res, 1)),
                dtype=torch.float32,
                device=device,
            )
        else:
            test_batch = prediction
        with torch.no_grad():
            prediction = model(test_batch)
            prediction_np = prediction.detach().cpu().numpy()
        pred_unif.append(prediction_np.squeeze())

    unif_label = labels_unif[0, 9:19].reshape(10, res, res)
    err = np.abs(pred_unif - unif_label)
    visual_animate(unif_label, pred_unif, err)


def _prepare_grid(grid, h, w, device):
    """Ensure grid is (H*W, 2) on device."""
    if grid.dim() == 3:
        grid_flat = grid.view(h * w, -1)
    else:
        grid_flat = grid
    return grid_flat.to(device)


@torch.no_grad()
def rollout_predict_torch(model, batch, grid, steps=None, device="cuda"):
    """Roll out a model for N steps and return prediction/target arrays.

    Expects batch["x"], batch["y"] shaped (B, T, H, W, 1, 1) or (B, H, W, 1, 1).
    Returns (preds, target) as numpy arrays with shape (steps, H, W) for sample 0.
    """
    model.eval()
    x = batch["x"].to(device).squeeze(-1).squeeze(-1)
    y = batch["y"].to(device).squeeze(-1).squeeze(-1)

    if x.dim() == 3:
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

    b, t, h, w = x.shape
    steps = t if steps is None else min(steps, t)

    grid_flat = _prepare_grid(grid, h, w, device)
    coords = grid_flat.unsqueeze(0).repeat(b, 1, 1)

    current = x[:, 0]
    preds = []
    for _ in range(steps):
        fx = current.view(b, h * w, 1)
        pred = model(coords, fx)
        pred_field = pred.view(b, h, w)
        preds.append(pred_field)
        current = pred_field

    preds = torch.stack(preds, dim=1)  # (B, steps, H, W)
    target = y[:, :steps]

    return preds[0].detach().cpu().numpy(), target[0].detach().cpu().numpy()


def visual_rollout_torch(model, batch, grid, steps=None, device="cuda"):
    """Visualize N-step rollout for a torch model and batch."""
    pred, target = rollout_predict_torch(model, batch, grid, steps=steps, device=device)
    err = np.abs(pred - target)
    visual_animate(target, pred, err)
