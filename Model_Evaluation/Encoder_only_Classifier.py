import torch
import torch.nn as nn
from transformers import XLNetModel, RobertaModel, DebertaModel, DebertaV2Model, BertModel

class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

class XLNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(XLNetClassifier, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.xlnet.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled_output)
        return self.fc(x)

class RoBERTaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)
    
# class DeBERTaClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(DeBERTaClassifier, self).__init__()
#         self.deberta = DebertaModel.from_pretrained('microsoft/deberta-v3-base')
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(self.deberta.config.hidden_size, num_classes)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.last_hidden_state[:, 0, :]
#         x = self.dropout(pooled_output)
#         return self.fc(x)
    
class DeBERTaV3Classifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.26):
        super(DeBERTaV3Classifier, self).__init__()
        self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.MultiheadAttention(embed_dim=self.deberta.config.hidden_size, num_heads=8)
        self.fc1 = nn.Linear(self.deberta.config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):    
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]

        # Transpose for attention layer: [seq_length, batch_size, hidden_size]
        sequence_output = sequence_output.permute(1, 0, 2)
        attn_output, _ = self.attention(sequence_output, sequence_output, sequence_output)
        # Transpose back: [batch_size, seq_length, hidden_size]
        attn_output = attn_output.permute(1, 0, 2)

        # Pooling
        pooled_output = torch.mean(attn_output, dim=1)
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
    
# class DeBERTaV3Classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(DeBERTaV3Classifier, self).__init__()
#         self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')
#         self.dropout = nn.Dropout(0.3)
#         self.fc = nn.Linear(self.deberta.config.hidden_size, num_classes)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.last_hidden_state[:, 0, :]
#         x = self.dropout(pooled_output)
#         return self.fc(x)

# class DeeperDeBERTaClassifier(nn.Module):
#     def __init__(self, num_classes, hidden_sizes=[768, 512, 256], dropout_rate=0.3):
#         super(DeeperDeBERTaClassifier, self).__init__()
#         self.deberta = DebertaModel.from_pretrained('microsoft/deberta-base')
        
#         layers = []
#         for i in range(len(hidden_sizes) - 1):
#             layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout_rate))
        
#         self.fc_layers = nn.Sequential(*layers)
#         self.classifier = nn.Linear(hidden_sizes[-1], num_classes)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.last_hidden_state[:, 0, :]
#         x = self.fc_layers(pooled_output)
#         return self.classifier(x)