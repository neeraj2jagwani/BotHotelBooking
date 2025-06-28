import spacy
import random
from spacy.training import Example
from spacy.util import minibatch, compounding
import json
import os
from datetime import datetime
import joblib

class NLUModel:
    def __init__(self, model_path=None):
        self.model_path = model_path or "models/nlu_model"
        self.nlp = None
        self.ner = None
        self.label_encoder = None
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create a new one"""
        if os.path.exists(self.model_path):
            self.nlp = spacy.load(self.model_path)
            print("Loaded existing model")
        else:
            self.nlp = spacy.blank('en')
            self.nlp.add_pipe('ner')
            print("Created new model")
    
    def prepare_training_data(self, intent_data):
        """Convert intent data to spaCy's training format"""
        TRAIN_DATA = []
        
        for intent_name, intent_data in intent_data.items():
            for pattern in intent_data.get('patterns', []):
                # Create entity annotations
                doc = self.nlp.make_doc(pattern)
                entities = []
                
                # Extract entities (simplified - in real app, use more sophisticated entity recognition)
                for ent in doc.ents:
                    entities.append((ent.start_char, ent.end_char, ent.label_))
                
                TRAIN_DATA.append((pattern, {"intent": intent_name, "entities": entities}))
        
        return TRAIN_DATA
    
    def train_model(self, training_data, n_iter=100):
        """Train the NER model"""
        # Add entity labels to the model
        ner = self.nlp.get_pipe('ner')
        
        # Add new entity labels
        for _, annotations in training_data:
            for ent in annotations.get('entities', []):
                ner.add_label(ent[2])
        
        # Disable other pipeline components during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            
            # Training loop
            for itn in range(n_iter):
                random.shuffle(training_data)
                losses = {}
                
                # Batch the examples and iterate over them
                batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        self.nlp.update([example], drop=0.5, losses=losses)
                
                print(f"Iteration {itn + 1}, Losses: {losses}")
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.nlp.to_disk(self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def predict_intent(self, text):
        """Predict the intent of a user message"""
        doc = self.nlp(text)
        
        # Extract entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # In a real implementation, you would use a classifier for intent
        # For simplicity, we'll use pattern matching here
        text_lower = text.lower()
        
        # Simple intent classification (can be replaced with a trained classifier)
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            intent = 'greeting'
        elif any(word in text_lower for word in ['book', 'reservation', 'reserve']):
            intent = 'book_room'
        elif any(word in text_lower for word in ['availability', 'available', 'vacancy']):
            intent = 'check_availability'
        elif any(word in text_lower for word in ['cancel', 'cancellation']):
            intent = 'cancel_booking'
        elif any(word in text_lower for word in ['thank', 'thanks']):
            intent = 'thanks'
        else:
            intent = 'fallback'
        
        return {
            'intent': intent,
            'entities': entities,
            'confidence': 0.9  # Placeholder confidence score
        }
    
    def extract_booking_details(self, text):
        """Extract booking details from user input"""
        doc = self.nlp(text)
        
        details = {
            'check_in': None,
            'check_out': None,
            'room_type': None,
            'guests': 1
        }
        
        # Extract dates (simplified - in a real app, use date parsing)
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                if not details['check_in']:
                    details['check_in'] = ent.text
                else:
                    details['check_out'] = ent.text
            
            if ent.label_ == 'ROOM_TYPE':
                details['room_type'] = ent.text
            
            if ent.label_ == 'CARDINAL':
                try:
                    num = int(ent.text)
                    if num > 0 and num <= 10:  # Reasonable guest number
                        details['guests'] = num
                except ValueError:
                    pass
        
        return details

# Example usage
if __name__ == "__main__":
    # Sample training data
    TRAIN_DATA = [
        ("I want to book a deluxe room", {"intent": "book_room", "entities": [(25, 37, "ROOM_TYPE")]}),
        ("Book me a standard room for 2 people", {"intent": "book_room", "entities": [(10, 19, "ROOM_TYPE"), (24, 25, "CARDINAL")]}),
        ("Do you have any suites available?", {"intent": "check_availability", "entities": [(18, 24, "ROOM_TYPE")]}),
        ("I'd like to cancel my booking", {"intent": "cancel_booking", "entities": []}),
        ("Hello, how are you?", {"intent": "greeting", "entities": []}),
    ]
    
    # Initialize and train the model
    nlu = NLUModel()
    nlu.train_model(TRAIN_DATA, n_iter=10)
    
    # Test the model
    test_queries = [
        "Can I book a deluxe room?",
        "Do you have any rooms available tomorrow?",
        "I need to cancel my reservation",
        "Hi there!"
    ]
    
    for query in test_queries:
        result = nlu.predict_intent(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {result['intent']}")
        print(f"Entities: {result['entities']}")
