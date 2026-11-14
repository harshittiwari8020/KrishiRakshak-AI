"""Train an EfficientNet-B0 transfer-learning classifier on ImageFolder data."""
from html import parser
from pathlib import Path
from pyexpat import model
import argparse, json, random, copy
from xml.parsers.expat import model
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
class SafeImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(
            entry.name
            for entry in Path(directory).iterdir()
            if entry.is_dir() and entry.name != "None"
        )

        if not classes:
            raise FileNotFoundError(f"No valid class folders found in {directory}")

        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        return classes, class_to_idx


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/plantvillage")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows for easiest setup.")
    args = parser.parse_args()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)

    train_dir, test_dir = Path(args.data)/"train", Path(args.data)/"test"
    if not train_dir.exists():
        raise FileNotFoundError(f"{train_dir} not found. Run: python download_dataset.py")
    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(18),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.12),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    ])
    base = SafeImageFolder(train_dir)
    classes = base.classes
    if len(classes) < 2: raise ValueError("Need at least two class folders in training data.")
    indices = np.arange(len(base))
    train_idx, val_idx = train_test_split(indices, test_size=.15, random_state=42,
                                          stratify=base.targets)
    train_ds = SafeImageFolder(train_dir, transform=train_tf)
    val_ds = SafeImageFolder(train_dir, transform=eval_tf)
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

# Freeze EfficientNet backbone
    for param in model.features.parameters():
         param.requires_grad = False

# Train only the classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=.4)
    best_acc = 0.0
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        model.features.eval()
        running_loss=0
        correct=0
        total=0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item()*labels.size(0)
            correct += (logits.argmax(1)==labels).sum().item(); total += labels.size(0)
        train_acc = correct/max(total,1)
        model.eval(); vcorrect=0; vtotal=0; preds=[]; actual=[]
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits=model(images); p=logits.argmax(1)
                vcorrect += (p==labels).sum().item(); vtotal += labels.size(0)
                preds.extend(p.cpu().tolist()); actual.extend(labels.cpu().tolist())
        val_acc=vcorrect/max(vtotal,1); scheduler.step(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs} | loss {running_loss/max(total,1):.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc=val_acc
            torch.save({"state_dict":model.state_dict(),"classes":classes,
                        "image_size":args.image_size,"architecture":"efficientnet_b0"},
                       "models/crop_disease_model.pth")

    # Optional untouched test split evaluation, if present.
    if test_dir.exists() and any(test_dir.iterdir()):
        test_ds=SafeImageFolder(test_dir, transform=eval_tf)
        if test_ds.classes != classes:
            print("WARNING: test class folders differ from train; skipping test evaluation.")
        else:
            loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
            model.load_state_dict(torch.load("models/crop_disease_model.pth",map_location=device,weights_only=False)["state_dict"])
            model.eval(); y_true=[]; y_pred=[]
            with torch.no_grad():
                for x,y in loader:
                    p=model(x.to(device)).argmax(1).cpu().tolist()
                    y_pred.extend(p); y_true.extend(y.tolist())
            report=classification_report(y_true,y_pred,target_names=classes,output_dict=True,zero_division=0)
            with open("outputs/test_classification_report.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
            print("Test accuracy:", round(accuracy_score(y_true,y_pred),4))
            print("Detailed report saved to outputs/test_classification_report.json")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print("Model saved: models/crop_disease_model.pth")

if __name__ == "__main__":
    main()
