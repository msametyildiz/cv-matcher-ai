"""
trainer.py

Bu modül, transformer tabanlı sınıflandırma modelinin eğitimini yapar.
Veri yükleme, tokenization, loss hesaplama ve model kaydetme işlemlerini içerir.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from cvmatcher.config import TRAINING_PARAMS, BEST_MODEL_PATH
from cvmatcher.model import CVMatcherModel
from cvmatcher.tokenizer import tokenizer
from cvmatcher.preprocessor import preprocess


class MatchDataset(Dataset):
    def __init__(self, texts1, texts2, labels):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels

    def __len__(self):
        return len(self.texts1)

    def __getitem__(self, idx):
        encoding = tokenizer(
            text=self.texts1[idx],
            text_pair=self.texts2[idx],
            padding="max_length",
            truncation=True,
            max_length=TRAINING_PARAMS["max_seq_length"],
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(float(self.labels[idx]), dtype=torch.float)
        }


from sklearn.metrics import accuracy_score

def train_model(df):
    # 1. Veriyi eğitim ve doğrulama olarak ayır
    X_train, X_val, y_train, y_val = train_test_split(
        df["cv_text"], df["label"], test_size=0.2, random_state=42
    )

    # 2. Ön işleme
    X_train = [preprocess(x) for x in X_train]
    X_val = [preprocess(x) for x in X_val]
    y_train = list(y_train)
    y_val = list(y_val)

    # 3. Dataset ve DataLoader
    job_text = df["job_text"].iloc[0]
    train_dataset = MatchDataset(X_train, [job_text] * len(X_train), y_train)
    val_dataset = MatchDataset(X_val, [job_text] * len(X_val), y_val)

    train_loader = DataLoader(train_dataset, batch_size=TRAINING_PARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_PARAMS["batch_size"])

    # 4. Model ve optimizer
    model = CVMatcherModel()
    optimizer = AdamW(model.parameters(), lr=TRAINING_PARAMS["learning_rate"])
    criterion = torch.nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc = 0.0
    for epoch in range(TRAINING_PARAMS["epochs"]):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds += outputs.detach().cpu().numpy().round().flatten().tolist()
            all_labels += labels.detach().cpu().numpy().flatten().tolist()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                val_preds += outputs.cpu().numpy().round().flatten().tolist()
                val_labels += labels.cpu().numpy().flatten().tolist()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)

        print(f"[Epoch {epoch+1}] 🔧 Train Loss: {avg_train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"[Epoch {epoch+1}] ✅ Val   Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f}")

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("💾 Yeni en iyi model kaydedildi!")

    print("🎉 Eğitim tamamlandı! En iyi doğruluk: {:.4f}".format(best_val_acc))

