import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import infer_num_classes, build_loaders
from .models import build_model

def train_one_epoch(model, loader, device, optimizer, scaler, criterion):
    model.train()
    total_loss, correct, count = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total_loss += loss.item() * x.size(0)
        count += x.size(0)
    return total_loss / count, correct / count

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, y in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total_loss += loss.item() * x.size(0)
        count += x.size(0)
    return total_loss / count, correct / count

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True, help="Dataset-B train dir")
    p.add_argument("--val_dir",   required=True, help="Dataset-B val dir")
    p.add_argument("--test_dir",  default=None, help="optional")
    p.add_argument("--phase1_ckpt", required=True, help="Path to phase1_<arch>_best.pt")
    p.add_argument("--arch", type=str, default=None,
                   help="Override arch; else load from ckpt. Choices: resnet50, efficientnet_b3, inception_v3")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine new class count from Dataset-B
    num_classes_B = infer_num_classes(args.train_dir)

    # Load Phase-1 checkpoint metadata
    ckpt = torch.load(args.phase1_ckpt, map_location="cpu")
    arch = (args.arch or ckpt.get("arch", "resnet50")).lower()

    # Build model for Dataset-B class count
    model_B, input_size = build_model(arch, num_classes=num_classes_B)

    # Build loaders with backbone-appropriate input size
    tr_loader, va_loader, te_loader = build_loaders(
        args.train_dir, args.val_dir, args.test_dir, image_size=input_size,
        batch_size=args.batch_size
    )

    # Build a temporary model with Phase-1 class count to load weights safely
    num_classes_A = ckpt.get("num_classes", None)
    if num_classes_A is None:
        raise SystemExit("Phase-1 checkpoint missing 'num_classes'. Re-train Phase-1 with this codebase.")
    temp_A, _ = build_model(arch, num_classes=num_classes_A)
    temp_A.load_state_dict(ckpt["state_dict"], strict=True)

    # Transfer backbone weights to the B model; keep new classifier
    missing = model_B.load_state_dict(temp_A.state_dict(), strict=False)
    print("Loaded Phase-1 weights into Phase-2 model. Missing/Unexpected keys:", missing)

    if args.freeze_backbone:
        for name, p in model_B.named_parameters():
            # keep final classifier trainable; freeze rest
            if ("fc" in name) or ("classifier" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

    model_B = model_B.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model_B.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model_B, tr_loader, device, optimizer, scaler, criterion)
        va_loss, va_acc = evaluate(model_B, va_loader, device, criterion)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")
        history.append({"epoch":epoch,"train_loss":tr_loss,"train_acc":tr_acc,"val_loss":va_loss,"val_acc":va_acc})

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "arch": arch,
                "state_dict": model_B.state_dict(),
                "num_classes": num_classes_B
            }, out_dir / f"phase2_{arch}_best.pt")
            print(f"Saved best Phase-2 ckpt (val_acc={best_val_acc:.4f})")

    with open(out_dir / f"phase2_{arch}_history.json","w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
