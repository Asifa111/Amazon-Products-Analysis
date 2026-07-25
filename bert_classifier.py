import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

class ProductDataset(Dataset):
    def __init__(self, descriptions, categories, tokenizer, max_length=128):
        self.descriptions = descriptions
        self.categories = categories
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create category to index mapping
        self.category_to_idx = {cat: idx for idx, cat in enumerate(sorted(set(categories)))}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        
        # Compute class weights
        category_indices = [self.category_to_idx[cat] for cat in categories]
        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(category_indices),
            y=category_indices
        )
        self.class_weights = torch.FloatTensor(self.class_weights)
        
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        description = str(self.descriptions[idx])
        category = self.categories[idx]
        category_idx = self.category_to_idx[category]
        
        # Tokenize the description
        encoding = self.tokenizer(
            description,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(category_idx, dtype=torch.long),
            'weight': self.class_weights[category_idx]
        }

class BERTClassifier:
    def __init__(self, num_labels):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.model.to(self.device)
        
    def get_embedding(self, text):
        """Get BERT embedding for a given text"""
        self.model.eval()
        with torch.no_grad():
            # Tokenize and prepare input
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get the last hidden state
            outputs = self.model.bert(**inputs)
            last_hidden_state = outputs.last_hidden_state
            
            # Use mean pooling over the last hidden state
            embedding = torch.mean(last_hidden_state, dim=1)
            return embedding.squeeze()
        
    def train(self, train_dataloader, val_dataloader, epochs=5, learning_rate=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,  # 10% of total steps for warmup
            num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            for batch in tqdm(train_dataloader, desc='Training'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                weights = batch['weight'].to(self.device)
                
                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Apply class weights to the loss
                loss = outputs.loss * weights.mean()
                train_loss += loss.item()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_steps += 1
            
            avg_train_loss = train_loss / train_steps
            print(f'Average training loss: {avg_train_loss}')
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_steps = 0
            correct_preds = 0
            total_preds = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc='Validation'):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    weights = batch['weight'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss * weights.mean()
                    val_loss += loss.item()
                    val_steps += 1
                    
                    # Calculate accuracy
                    preds = torch.argmax(outputs.logits, dim=1)
                    correct_preds += (preds == labels).sum().item()
                    total_preds += len(labels)
            
            avg_val_loss = val_loss / val_steps
            accuracy = correct_preds / total_preds
            print(f'Average validation loss: {avg_val_loss}')
            print(f'Validation accuracy: {accuracy:.4f}')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), 'best_bert_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    # Load the best model
                    self.model.load_state_dict(torch.load('best_bert_model.pt'))
                    break
    
    def predict(self, description):
        self.model.eval()
        
        encoding = self.tokenizer(
            description,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence

def train_model():
    # Load the data
    df = pd.read_csv('product.csv')
    
    # Create dataset
    dataset = ProductDataset(
        descriptions=df['Description'].values,
        categories=df['Categories'].values,
        tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    )
    
    # Split the data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Description'].values,
        df['Categories'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Create datasets
    train_dataset = ProductDataset(train_texts, train_labels, dataset.tokenizer)
    val_dataset = ProductDataset(val_texts, val_labels, dataset.tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize classifier
    classifier = BERTClassifier(num_labels=len(dataset.category_to_idx))
    
    # Train the model
    classifier.train(train_dataloader, val_dataloader, epochs=5)
    
    # Save the model
    torch.save(classifier.model.state_dict(), 'bert_product_classifier.pt')
    
    # Save category mappings
    category_mappings = {
        'idx_to_category': dataset.idx_to_category,
        'category_to_idx': dataset.category_to_idx
    }
    torch.save(category_mappings, 'category_mappings.pt')
    
    return classifier, dataset

def predict_category(description, classifier, dataset):
    predicted_class, confidence = classifier.predict(description)
    category = dataset.idx_to_category[predicted_class]
    return category, confidence

if __name__ == '__main__':
    # Train the model
    classifier, dataset = train_model()
    
    # Test the model with different categories
    test_descriptions = [
        "A thrilling mystery novel with unexpected twists and turns",
        "A powerful gaming laptop with RTX graphics and high refresh rate display",
        "Premium running shoes with advanced cushioning technology"
    ]
    
    print("\nTesting predictions:")
    print("-" * 50)
    
    for description in test_descriptions:
        category, confidence = predict_category(description, classifier, dataset)
        print(f"\nDescription: {description}")
        print(f"Predicted category: {category}")
        print(f"Confidence: {confidence:.2f}")
        print("-" * 50) 