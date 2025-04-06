"""
model.py

Hugging Face transformer modelleri (BERT, RoBERTa vb.) üzerine
basit bir sınıflandırma katmanı ekleyerek CV ve iş tanımı eşleşmesini
ikili sınıflandırma problemi olarak çözer.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from cvmatcher.config import MODEL_NAME

class CVMatcherModel(nn.Module):
    def __init__(self):
        super(CVMatcherModel, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)  # Binary classification

    def forward(self, input_ids, attention_mask):
        """
        Modelin ileriye doğru geçişi (forward pass)

        Args:
            input_ids (torch.Tensor): Token ID'leri
            attention_mask (torch.Tensor): Dikkat maskesi

        Returns:
            torch.Tensor: [0, 1] arası olasılık çıktısı
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token'ın çıktısı
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        prob = torch.sigmoid(logits)  # sigmoid -> 0-1 arası
        return prob
