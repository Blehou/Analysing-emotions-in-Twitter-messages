import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset as HFDataset

## Loading the dataset
file_path = r"C:\Jean Eudes Folder\_Projects\Generative_AI_LLM\Emotions_Datasets\text.csv"
df = pd.read_csv(file_path)
assert "text" in df.columns and "label" in df.columns, "Error: 'text' and 'label' columns are required"
df.dropna(inplace=True)

## Tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Convert to Hugging Face Dataset
dataset = HFDataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Split dataset
train_size = int(0.8 * len(tokenized_dataset))
val_size = int(0.1 * len(tokenized_dataset))
test_size = len(tokenized_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(tokenized_dataset, [train_size, val_size, test_size])

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Training arguments
training_args = TrainingArguments(
    output_dir=r"C:\Jean Eudes Folder\_Projects\Generative_AI_LLM\Emotions_Datasets\results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=r"C:\Jean Eudes Folder\_Projects\Generative_AI_LLM\Emotions_Datasets\logs",
    logging_steps=10,
    push_to_hub=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Evaluation
preds = trainer.predict(test_dataset)
pred_labels = np.argmax(preds.predictions, axis=1)
true_labels = preds.label_ids
print("Test set accuracy:", accuracy_score(true_labels, pred_labels))
print("Classification report:\n", classification_report(true_labels, pred_labels))

# Inference
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    labels_map = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
    return labels_map[prediction]

# Example usage
new_text = "I feel amazing today!"
predicted_emotion = predict_emotion(new_text)
print(f"Text: {new_text}\nPrediction: {predicted_emotion}")
