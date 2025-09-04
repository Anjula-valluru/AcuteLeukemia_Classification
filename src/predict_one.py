import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from .models import build_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    arch = ckpt.get("arch", "resnet50")
    num_classes = ckpt.get("num_classes", 2)

    model, input_size = build_model(arch, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    img = Image.open(Path(args.image)).convert("RGB")
    x = tfms(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
    print("Probabilities:", probs)

if __name__ == "__main__":
    main()
