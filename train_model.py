import json
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import os
from pathlib import Path
import joblib

class HotelNLPModel:
    def __init__(self, model_name='en_core_web_sm'):
        """Initialize the NLP model"""
        self.model_name = model_name
        self.nlp = None
        self.model_dir = Path("models/nlp_model")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoder = {}
        
    def load_or_create_model(self):
        """Load existing model or create a new one"""
        model_path = self.model_dir / "nlp_model"
        
        if model_path.exists():
            print("Loading existing model...")
            self.nlp = spacy.load(model_path)
        else:
            print("Creating new model...")
            self.nlp = spacy.blank('en')
            
            # Add NER pipeline
            if 'ner' not in self.nlp.pipe_names:
                self.nlp.add_pipe('ner')
            
            # Add text categorizer for intent classification
            if 'textcat' not in self.nlp.pipe_names:
                textcat = self.nlp.add_pipe('textcat')
            else:
                textcat = self.nlp.get_pipe('textcat')
            
            # Add labels for text classification (intents)
            self.label_encoder = self._create_label_encoder()
            for label in self.label_encoder.keys():
                # Ensure label is a string
                label_str = str(label)
                textcat.add_label(label_str)
    
    def _create_label_encoder(self):
        """Create a mapping between intent names and numeric labels"""
        with open('training/train_intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)
        
        # Ensure we're using string keys and values
        return {str(intent): int(idx) for idx, intent in enumerate(intents.keys())}
    
    def _prepare_training_data(self):
        """Prepare training data in spaCy format"""
        with open('training/train_intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)
        
        train_data = []
        
        for intent_name, intent_data in intents.items():
            intent_name = str(intent_name)  # Ensure intent_name is a string
            for pattern in intent_data.get('patterns', []):
                pattern = str(pattern)  # Ensure pattern is a string
                # Create entity annotations
                doc = self.nlp.make_doc(pattern)
                entities = []
                
                # Simple entity recognition (can be enhanced)
                # This is a simplified version - in a real app, you'd want more sophisticated entity recognition
                if 'room_type' in intent_data.get('entities', {}):
                    for room_type in intent_data['entities']['room_type']:
                        if room_type in pattern.lower():
                            start = pattern.lower().find(room_type)
                            end = start + len(room_type)
                            entities.append((start, end, 'ROOM_TYPE'))
                
                # Create training example
                train_example = {
                    'text': pattern,
                    'intent': intent_name,
                    'entities': entities
                }
                train_data.append(train_example)
        
        return train_data
    
    def train(self, n_iter=30):
        """Train the NLP model"""
        train_data = self._prepare_training_data()
        print(f"Training on {len(train_data)} examples")
        
        # Disable other pipeline components during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in ['ner', 'textcat']]
        with self.nlp.disable_pipes(*other_pipes):
            # Initialize the model with random weights
            optimizer = self.nlp.begin_training()
            
            # Training loop
            print("Training the model...")
            best_accuracy = 0
            
            for i in range(n_iter):
                losses = {}
                random.shuffle(train_data)
                
                # Split into training and evaluation sets (80/20)
                split = int(0.8 * len(train_data))
                train_examples = train_data[:split]
                eval_examples = train_data[split:]
                
                # Training
                batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    examples = []
                    for example in batch:
                        doc = self.nlp.make_doc(example['text'])
                        
                        # Create entity annotations
                        entities = []
                        for start, end, label in example.get('entities', []):
                            span = doc.char_span(start, end, label=label)
                            if span is not None:
                                entities.append(span)
                        
                        # Create example for text classification
                        cats = {intent: 0.0 for intent in self.label_encoder}
                        cats[example['intent']] = 1.0
                        
                        example = Example.from_dict(doc, {
                            'entities': [(e.start_char, e.end_char, e.label_) for e in entities],
                            'cats': cats
                        })
                        examples.append(example)
                    
                    # Update the model
                    self.nlp.update(examples, drop=0.3, losses=losses, sgd=optimizer)
                
                # Evaluate
                correct = 0
                for example in eval_examples:
                    doc = self.nlp(example['text'])
                    if hasattr(doc, 'cats') and doc.cats:
                        predicted = max(doc.cats.items(), key=lambda x: x[1])
                        if predicted[0] == example['intent'] and predicted[1] > 0.5:
                            correct += 1
                
                accuracy = correct / len(eval_examples) if eval_examples else 0
                print(f"Epoch {i+1}, Losses: {losses}, Accuracy: {accuracy:.2f}")
                
                # Save the best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    self.nlp.to_disk(self.model_dir / "nlp_model")
                    print(f"New best model saved with accuracy: {accuracy:.2f}")
        
        # Ensure we have at least one model saved
        if not (self.model_dir / "nlp_model").exists():
            self.nlp.to_disk(self.model_dir / "nlp_model")
            
        print(f"Training complete. Best accuracy: {best_accuracy:.2f}")
        print(f"Model saved to {self.model_dir}")
    
    def predict(self, text):
        """Predict intent and entities from text"""
        doc = self.nlp(text)
        
        # Get the most likely intent
        intents = [(label, score) for label, score in doc.cats.items()]
        intent = max(intents, key=lambda x: x[1]) if intents else ("fallback", 0.0)
        
        # Get entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        return {
            'text': text,
            'intent': intent[0],
            'confidence': float(intent[1]),
            'entities': entities
        }

def main():
    # Initialize and train the model
    nlp_model = HotelNLPModel()
    nlp_model.load_or_create_model()
    nlp_model.train(n_iter=10)
    
    # Test the model with some example queries
    test_queries = [
        "I'd like to book a deluxe room for next weekend",
        "What time is check-out?",
        "Do you have any rooms available tomorrow?",
        "How much does a suite cost?",
        "I need to cancel my reservation"
    ]
    
    for query in test_queries:
        result = nlp_model.predict(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        if result['entities']:
            print("Entities:")
            for entity, label in result['entities']:
                print(f"  - {entity} ({label})")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models/nlp_model", exist_ok=True)
    os.makedirs("training", exist_ok=True)
    
    # Download the English language model if not already present
    if not spacy.util.is_package("en_core_web_sm"):
        print("Downloading English language model...")
        spacy.cli.download("en_core_web_sm")
    
    main()
