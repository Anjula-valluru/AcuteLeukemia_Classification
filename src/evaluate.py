import argparse
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from .models import build_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    arch = ckpt.get("arch", "resnet50")
    num_classes = ckpt.get("num_classes", None)
    if num_classes is None:
        # infer from dataset if not stored
        tmp = datasets.ImageFolder(Path(args.test_dir))
        num_classes = len(tmp.classes)

    _, input_size = build_model(arch, num_classes=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ds = datasets.ImageFolder(Path(args.test_dir), transform=tfms)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model, _ = build_model(arch, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()

    all_preds, all_tgts = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_tgts.extend(y.tolist())

    print("Classes:", ds.classes)
    print(classification_report(all_tgts, all_preds, target_names=ds.classes, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(all_tgts, all_preds))

if __name__ == "__main__":
    main()
