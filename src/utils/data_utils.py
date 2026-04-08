import torch


def make_grid(h, w, device, dtype):
    ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    return grid.view(-1, 2)


def prepare_batch(batch, grid, device="cuda"):
    x = batch["x"].to(device).squeeze(-1).squeeze(-1)
    y = batch["y"].to(device).squeeze(-1).squeeze(-1)

    while x.dim() > 3:
        x = x[:, 0]
        y = y[:, 0]

    if x.dim() != 3:
        raise ValueError(f"Expected x to have 3 dims (B,H,W), got {tuple(x.shape)}")

    b, h, w = x.shape
    grid_flat = grid.view(h * w, -1) if grid.dim() == 3 else grid

    coords = grid_flat.to(device).unsqueeze(0).repeat(b, 1, 1)
    fx = x.view(b, h * w, 1)
    target = y.view(b, h * w, 1)
    return coords, fx, target
