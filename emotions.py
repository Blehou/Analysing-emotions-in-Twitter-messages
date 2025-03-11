import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset as HFDataset

## Chargement du dataset
file_path = r"C:\Jean Eudes Folder\_Projects\Generative_AI_LLM\Emotions_Datasets\text.csv"
df = pd.read_csv(file_path)

# Vérification des colonnes
assert "text" in df.columns and "label" in df.columns, "Erreur : les colonnes 'text' et 'label' sont requises"

# Suppression des valeurs manquantes
df.dropna(inplace=True)

## Prétraitement des données
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Utilisation de EOS comme padding

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Conversion en Dataset Hugging Face
dataset = HFDataset.from_pandas(df)

# Tokenization avant la séparation
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Séparation des données
train_size = int(0.8 * len(tokenized_dataset))
val_size = int(0.1 * len(tokenized_dataset))
test_size = len(tokenized_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(tokenized_dataset, [train_size, val_size, test_size])

## Modèle GPT-2 pour classification
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=6)
model.config.pad_token_id = model.config.eos_token_id  # Correction du padding

## Entraînement du modèle
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()

## Évaluation du modèle
preds = trainer.predict(test_dataset)
pred_labels = np.argmax(preds.predictions, axis=1)
true_labels = preds.label_ids

print("Accuracy sur le test set:", accuracy_score(true_labels, pred_labels))
print("Rapport de classification:\n", classification_report(true_labels, pred_labels))

## Prédiction sur de nouveaux textes
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    labels_map = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
    return labels_map[prediction]

# Exemple d'utilisation
new_text = "I feel amazing today!"
predicted_emotion = predict_emotion(new_text)
print(f"Texte: {new_text}\nPrédiction: {predicted_emotion}")