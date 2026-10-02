"""Evaluator classifiers (fine / make labels). EVALUATION ONLY: no method may load or condition on them."""
import copy

import timm
import torch
import torch.nn.functional as F
from PIL import Image


class LabelledImages(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB")), self.labels[i]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = [(model(x.to(device)).float().cpu(), y) for x, y in loader]
    return torch.cat([o for o, _ in out]), torch.cat([torch.as_tensor(y) for _, y in out])


def train_classifier(arch, n_classes, train_dl, val_dl, device, epochs, lr, weight_decay):
    """Fine-tune from ImageNet init; keep the epoch with the best val accuracy."""
    model = timm.create_model(arch, pretrained=True, num_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(train_dl))
    scaler, best, history = torch.amp.GradScaler(), (-1, None), []
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        logits, y = predict(model, val_dl, device)
        acc = (logits.argmax(1) == y).float().mean().item()
        history.append({"epoch": epoch, "train_loss": loss.item(), "val_acc": acc})
        if acc > best[0]:
            best = (acc, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1])
    return model, best[0], history


def fit_temperature(logits, labels):
    """Temperature minimising NLL (fit on val only)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def ece(probs, labels, n_bins=15):
    """Expected calibration error with equal-width confidence bins."""
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    edges, err = torch.linspace(0, 1, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.float().mean().item() * abs(correct[m].mean().item() - conf[m].mean().item())
    return err


def load_classifier(path, device):
    """Evaluator checkpoint written by scripts/train_eval_classifier.py -> (model, temperature)."""
    ck = torch.load(path, map_location="cpu")
    model = timm.create_model(ck["arch"], pretrained=False, num_classes=ck["n_classes"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["temperature"]


@torch.no_grad()
def calibrated_probs(model, temperature, paths, transform, device, batch_size=128, num_workers=8):
    """Temperature-scaled class probabilities (n, K) for image files, rows in the order of `paths`."""
    dl = torch.utils.data.DataLoader(LabelledImages(paths, [0] * len(paths), transform), batch_size, num_workers=num_workers)
    return torch.cat([(model(x.to(device)).float() / temperature).softmax(1).cpu() for x, _ in dl]).numpy()
