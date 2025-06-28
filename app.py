from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import mysql.connector
from datetime import datetime, timedelta
import json
import os
import random
import re
import spacy
from pathlib import Path

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.urandom(24)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database configuration
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',  # Update with your MySQL username
        password='root',  # Update with your MySQL password
        database='hotel_booking_db'
    )

# Initialize database tables
def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            guest_name VARCHAR(100) NOT NULL,
            email VARCHAR(100) NOT NULL,
            check_in DATE NOT NULL,
            check_out DATE NOT NULL,
            room_type VARCHAR(50) NOT NULL,
            guests INT NOT NULL,
            status VARCHAR(20) DEFAULT 'confirmed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.close()
    conn.close()

# Initialize database
init_db()

# Load the NLP model
class HotelChatbot:
    def __init__(self):
        self.nlp = self._load_nlp_model()
        self.label_encoder = self._load_label_encoder()
    
    def _load_nlp_model(self):
        model_path = Path("models/nlp_model/nlp_model")
        if model_path.exists():
            print("Loading existing NLP model...")
            try:
                nlp = spacy.load(model_path)
                print(f"[DEBUG] Successfully loaded model from {model_path}")
                print(f"[DEBUG] Model pipeline: {nlp.pipe_names}")
                if 'textcat' in nlp.pipe_names:
                    textcat = nlp.get_pipe('textcat')
                    print(f"[DEBUG] TextCat labels: {textcat.labels}")
                return nlp
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                print("Falling back to blank model...")
        
        print("No trained model found or error loading. Using blank English model.")
        nlp = spacy.blank('en')
        if 'ner' not in nlp.pipe_names:
            nlp.add_pipe('ner')
        if 'textcat' not in nlp.pipe_names:
            textcat = nlp.add_pipe('textcat')
            # Add default labels to prevent errors
            textcat.add_label('greeting')
            textcat.add_label('fallback')
        return nlp
    
    def _load_label_encoder(self):
        try:
            with open('training/train_intents.json', 'r') as f:
                intents = json.load(f)
            return {intent: idx for idx, intent in enumerate(intents.keys())}
        except FileNotFoundError:
            print("Warning: train_intents.json not found. Using empty label encoder.")
            return {}
    
    def process_message(self, text, context=None):
        """Process user message and return intent and entities"""
        print(f"[DEBUG] Processing message: {text}")
        
        # Check for facility-related queries first
        facility_keywords = [
            'swimming pool', 'pool', 'gym', 'spa', 'restaurant', 
            'business center', 'room service', 'bar', 'laundry', 
            'concierge', 'wifi', 'mini-bar', 'minibar', 
            'air conditioning', 'tv', 'safe', 'work desk', 
            'coffee maker', 'hairdryer', 'iron', 'fridge'
        ]
        
        # Check for any facility mention in the message
        mentioned_facilities = [f for f in facility_keywords if f in text.lower()]
        
        if mentioned_facilities:
            print(f"[DEBUG] Detected facility query - forcing specific_facility intent")
            return {
                'intent': 'specific_facility',
                'entities': {'facility': mentioned_facilities},
                'original_text': text,
                'confidence': 0.95
            }
        
        # Process the text with spaCy
        doc = self.nlp(text.lower())
        
        # Get intent predictions
        intents = []
        if hasattr(doc, 'cats') and doc.cats:
            intents = [(label, float(score)) for label, score in doc.cats.items()]
            intents.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
            print(f"[DEBUG] Top 3 intents:")
            for i, (intent_name, score) in enumerate(intents[:3], 1):
                print(f"  {i}. {intent_name}: {score:.4f}")
        
        # Initialize intent and score
        intent = 'fallback'
        score = 0.0
        if intents:
            intent, score = intents[0]
        
        # Initialize entities
        entities = {}
        text_lower = text.lower()
        
        # Facility and amenity handling
        facility_keywords = [
            'swimming pool', 'pool', 'gym', 'spa', 'restaurant', 'business center', 'room service',
            'bar', 'laundry', 'concierge', 'wifi', 'mini-bar', 'minibar', 'air conditioning', 'tv',
            'safe', 'work desk', 'coffee maker', 'hairdryer', 'iron', 'fridge'
        ]
        
        # Check for specific facility/amenity mentions
        mentioned_facilities = [f for f in facility_keywords if f in text_lower]
        
        # Check for facility/amenity patterns
        facility_patterns = [
            'do you have', 'is there a', 'do you offer', 'can i use', 'do you provide',
            'is there', 'do you serve', 'are there', 'do your rooms have', 'is there access to',
            'can i find', 'is there', 'do you have', 'can i use', 'is there', 'do you have',
            'is there', 'do you offer', 'can i use', 'do you have', 'is there', 'do you have'
        ]
        
        has_facility_pattern = any(pattern in text_lower for pattern in facility_patterns)
        
        # If the query is about facilities/amenities and not about booking/availability
        booking_keywords = ['book', 'reserve', 'available', 'check', 'price', 'cost', 'how much', 'rate', 'rates']
        has_booking_context = any(keyword in text_lower for keyword in booking_keywords)
        
        if (mentioned_facilities or has_facility_pattern) and not has_booking_context:
            print(f"[DEBUG] Facility/amenity query detected, forcing specific_facility intent. Mentioned: {mentioned_facilities}")
            intent = 'specific_facility'
            
            # Add mentioned facilities to entities
            if mentioned_facilities:
                entities['facility'] = mentioned_facilities
            else:
                # Try to extract facility from the message
                for word in text_lower.split():
                    if word in facility_keywords and 'facility' not in entities:
                        entities['facility'] = [word]
                        break
        
        # Price-related queries
        price_keywords = ['price', 'cost', 'rate', 'how much', 'what\'s the price']
        if any(keyword in text_lower for keyword in price_keywords):
            print("[DEBUG] Price-related query detected, overriding intent to 'price_inquiry'")
            intent = 'price_inquiry'
        
        # Extract entities using spaCy
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME', 'CARDINAL', 'QUANTITY', 'MONEY']:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
        
        # Extract room types
        room_types = ['standard', 'deluxe', 'suite', 'executive', 'family']
        for room in room_types:
            if room in text_lower:
                if 'ROOM_TYPE' not in entities:
                    entities['ROOM_TYPE'] = []
                if room not in entities['ROOM_TYPE']:
                    entities['ROOM_TYPE'].append(room)
        
        # Extract number of guests
        if 'ROOM_TYPE' not in entities:
            room_types = ['standard', 'deluxe', 'suite']
            found_types = [rt for rt in room_types if rt in text.lower()]
            if found_types:
                entities['ROOM_TYPE'] = found_types
        
        # Confidence threshold check
        if intents and (score < 0.5 or (len(intents) > 1 and (score - intents[1][1]) < 0.2)):
            print(f"[DEBUG] Low confidence or small margin, using fallback. Top score: {score:.4f}, Next: {intents[1][1] if len(intents) > 1 else 0:.4f}")
            intent = 'fallback'
        
        result = {
            'intent': intent,
            'entities': entities,
            'original_text': text,
            'confidence': float(score)
        }
        print(f"[DEBUG] Final result: {result}")
        return result

# Initialize the chatbot
chatbot = HotelChatbot()

# Utility functions
def format_date(date_str):
    """Convert various date formats to YYYY-MM-DD"""
    if not date_str:
        return None
        
    # Try different date formats
    date_formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%m-%d-%Y',
        '%B %d, %Y',
        '%b %d, %Y',
        '%d %B %Y',
        '%d %b %Y'
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return None

def get_response(intent, entities=None, user_message=''):
    """Get response based on intent and entities from train_intents.json"""
    print(f"[DEBUG] Getting response for intent: {intent}")
    print(f"[DEBUG] Entities: {entities}")
    print(f"[DEBUG] User message: {user_message}")
    
    # Initialize entities if None
    if entities is None:
        entities = {}
    
    # For facility-related queries
    if intent in ['room_amenities', 'specific_facility']:
        print(f"[DEBUG] Handling facility-related query for intent: {intent}")
        user_message_lower = user_message.lower()
        
        # Load facilities from training data slots
        try:
            with open('training/train_intents.json', 'r', encoding='utf-8') as f:
                intents_data = json.load(f)
            all_facilities = intents_data.get('specific_facility', {}).get('slots', {}).get('facility', [])
            print(f"[DEBUG] Loaded facilities from training data: {all_facilities}")
        except Exception as e:
            print(f"[ERROR] Error loading facilities from training data: {e}")
            # Fallback list if loading fails
            all_facilities = [
                'swimming pool', 'pool', 'gym', 'fitness center', 'spa',
                'restaurant', 'business center', 'room service', 'bar',
                'laundry service', 'concierge'
            ]
        
        # Check if this is a general amenities query (no specific facility mentioned)
        is_general_query = any(phrase in user_message_lower for phrase in [
            'what amenities', 'what facilities', 'what do you offer', 
            'what services', 'what do you have', 'tell me about amenities',
            'list of amenities', 'amenities available', 'facilities available'
        ])
        
        # If it's a general query, return the general amenities list from training data
        if is_general_query:
            # Get the first response from training data which is our general amenities list
            with open('training/train_intents.json', 'r', encoding='utf-8') as f:
                intents_data = json.load(f)
            responses = intents_data.get('room_amenities', {}).get('responses', [])
            if responses:
                return responses[0]
            # Fallback response in case the training data is not available
            return """We offer a variety of amenities including a swimming pool, fitness center, restaurant, spa, and more. 
Would you like more information about any specific amenity?

You can ask about:
- Swimming pool hours
- Fitness center facilities
- Restaurant menu and timings
- Spa services
- Business center
- Room service
- And more!"""
        
        # If not a general query, check for specific facilities
        specific_facility = None
        
        # First, check entities from NLP
        if 'facility' in entities and entities['facility']:
            specific_facility = entities['facility'][0].lower()
            print(f"[DEBUG] Found facility in entities: {specific_facility}")
        
        # If no entity found, try to match from the message
        if not specific_facility:
            print("[DEBUG] No facility in entities, checking message text")
            # Check for exact matches first
            for facility in all_facilities:
                if facility in user_message_lower:
                    specific_facility = facility
                    print(f"[DEBUG] Found facility in message: {specific_facility}")
                    break
            
            # If still not found, try partial matches
            if not specific_facility:
                for facility in all_facilities:
                    if any(word in user_message_lower.split() for word in facility.split()):
                        specific_facility = facility
                        print(f"[DEBUG] Found partial facility match: {specific_facility}")
                        break
            
            # If still not found, check for 'pool' specifically
            if not specific_facility and 'pool' in user_message_lower:
                specific_facility = 'swimming pool'
                print(f"[DEBUG] Defaulting to 'swimming pool' for pool query")
        
        # If we have a specific facility, get response from training data
        if specific_facility:
            print(f"[DEBUG] Processing specific facility: {specific_facility}")
            try:
                with open('training/train_intents.json', 'r', encoding='utf-8') as f:
                    intents_data = json.load(f)
                
                # Get responses for specific_facility intent
                if 'specific_facility' in intents_data and 'responses' in intents_data['specific_facility']:
                    responses = intents_data['specific_facility']['responses']
                    if responses:
                        # Select a response and ensure facility name is properly formatted
                        response = random.choice(responses)
                        print(f"[DEBUG] Selected response template: {response}")
                        
                        # Replace placeholders if any
                        if 'slots' in intents_data['specific_facility']:
                            slots = intents_data['specific_facility']['slots']
                            print(f"[DEBUG] Available slots: {slots}")
                            
                            # Always use the facility from the message if we found one
                            if specific_facility:
                                response = response.replace('[facility]', specific_facility)
                                print(f"[DEBUG] Replaced [facility] with: {specific_facility}")
                            
                            # Replace other placeholders if they exist
                            for slot_type, slot_values in slots.items():
                                if slot_type != 'facility' and slot_values:  # Skip facility as we already handled it
                                    slot_value = random.choice(slot_values)
                                    response = response.replace(f'[{slot_type}]', slot_value)
                                    print(f"[DEBUG] Replaced [{slot_type}] with: {slot_value}")
                        
                        return response
                
                # Fallback if no specific response found
                return f"Yes, we have {specific_facility} available for our guests. Is there anything specific you'd like to know about it?"
                
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                print(f"[ERROR] Error loading facility responses: {e}")
                return f"Yes, we have {specific_facility} available for our guests. Is there anything specific you'd like to know about it?"
    
    # Enhanced fallback responses
    fallback_responses = [
        "I'm sorry, I'm not sure I understand. Could you rephrase your question?",
        "I'm still learning. Could you try asking in a different way?",
        "I want to make sure I help you correctly. Could you provide more details?",
        "I'm not quite sure how to respond to that. Here's what I can help with:"
        "\n- Room bookings and availability\n- Room rates and pricing\n- Hotel amenities\n- Cancellation policies",
        "I'm here to help with hotel bookings and information. Could you tell me more about what you need?"
    ]
    
    try:
        # Load responses from training data
        with open('training/train_intents.json', 'r', encoding='utf-8') as f:
            intents_data = json.load(f)
        
        print(f"[DEBUG] Available intents in training data: {list(intents_data.keys())}")
        
        # If no intent matched, return a fallback response
        if intent not in intents_data:
            print(f"[DEBUG] Intent '{intent}' not found in training data")
            return random.choice(fallback_responses)
            
        print(f"[DEBUG] Found intent '{intent}' in training data")
        
        # Get responses for the detected intent
        if 'responses' in intents_data[intent]:
            responses = intents_data[intent]['responses']
            print(f"[DEBUG] Found {len(responses)} responses for intent '{intent}'")
            if responses:
                response = random.choice(responses)
                print(f"[DEBUG] Selected response: {response}")
                
                # Replace any slot placeholders in the response
                if 'slots' in intents_data[intent]:
                    for slot_type, slot_values in intents_data[intent]['slots'].items():
                        if slot_type in entities and entities[slot_type]:
                            # Use the first entity of this type found
                            slot_value = entities[slot_type][0]
                            response = response.replace(f'[{slot_type}]', slot_value)
                        else:
                            # If no entity provided, pick a random value from the slot options
                            if slot_values:  # Check if there are any values to choose from
                                slot_value = random.choice(slot_values)
                                response = response.replace(f'[{slot_type}]', slot_value)
                
                return response
        
        # If no responses found for the intent, return a fallback
        print(f"[DEBUG] No responses found for intent '{intent}'")
        return random.choice(fallback_responses)
                
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] Error loading responses: {e}")
        return "I'm having trouble accessing my knowledge base. Please try again in a moment."
    
def extract_booking_details(entities, text):
    """Extract booking details from entities and text"""
    details = {
        'check_in': None,
        'check_out': None,
        'room_type': None,
        'guests': 1
    }
    
    # Try to extract dates using regex first (for suggested date formats)
    date_range_match = re.search(r'(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})', text)
    if date_range_match:
        check_in = date_range_match.group(1)
        check_out = date_range_match.group(2)
        if format_date(check_in) and format_date(check_out):
            details['check_in'] = format_date(check_in)
            details['check_out'] = format_date(check_out)
    
    # Extract from entities if not found in text
    if not details['check_in'] and 'DATE' in entities and entities['DATE']:
        dates = [format_date(d) for d in entities['DATE'] if format_date(d)]
        dates = sorted(list(set(dates)))  # Remove duplicates and sort
        
        if len(dates) >= 2:
            details['check_in'] = dates[0]
            details['check_out'] = dates[1]
        elif dates:
            details['check_in'] = dates[0]
            # Default to next day if only one date provided
            try:
                check_in = datetime.strptime(details['check_in'], '%Y-%m-%d')
                details['check_out'] = (check_in + timedelta(days=1)).strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                pass
    
    # Extract room type
    if 'ROOM_TYPE' in entities and entities['ROOM_TYPE']:
        details['room_type'] = entities['ROOM_TYPE'][0].capitalize()
    
    # Extract number of guests
    if 'CARDINAL' in entities and entities['CARDINAL']:
        try:
            guests = int(entities['CARDINAL'][0])
            if 1 <= guests <= 10:  # Reasonable guest number
                details['guests'] = guests
        except (ValueError, TypeError):
            pass
    
    # If we have a room type in the text but not in entities
    if not details['room_type']:
        room_types = ['standard', 'deluxe', 'suite']
        for rt in room_types:
            if rt in text.lower():
                details['room_type'] = rt.capitalize()
                break
    
    return details

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'hotel_booking_db'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS bookings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        guest_name VARCHAR(100) NOT NULL,
        check_in DATE NOT NULL,
        check_out DATE NOT NULL,
        room_type VARCHAR(50) NOT NULL,
        status VARCHAR(20) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Insert sample data if the table is empty
    cursor.execute("SELECT COUNT(*) FROM bookings")
    if cursor.fetchone()[0] == 0:
        sample_bookings = [
            ("John Doe", "2025-07-01", "2025-07-05", "Deluxe", "confirmed"),
            ("Jane Smith", "2025-07-10", "2025-07-15", "Suite", "pending")
        ]
        cursor.executemany('''
            INSERT INTO bookings (guest_name, check_in, check_out, room_type, status)
            VALUES (%s, %s, %s, %s, %s)
        ''', sample_bookings)
    
    conn.commit()
    cursor.close()
    conn.close()

# Routes
@app.route('/')
def home():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files from the static directory"""
    response = send_from_directory('static', filename)
    # Add headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Room prices per night
ROOM_PRICES = {
    'Standard': 100,
    'Deluxe': 150,
    'Suite': 250
}

def calculate_total_price(room_type, check_in, check_out):
    """Calculate total price for a booking"""
    if not all([room_type, check_in, check_out]):
        return 0
        
    try:
        check_in_date = datetime.strptime(check_in, '%Y-%m-%d')
        check_out_date = datetime.strptime(check_out, '%Y-%m-%d')
        nights = (check_out_date - check_in_date).days
        if nights <= 0:
            return 0
        return ROOM_PRICES.get(room_type, 0) * nights
    except (ValueError, TypeError):
        return 0

# Helper function to extract room type
def extract_room_type(text):
    text = text.lower()
    if 'deluxe' in text:
        return 'Deluxe'
    elif 'suite' in text:
        return 'Suite'
    return 'Standard'  # Default to Standard

# Track conversation state and history
conversation_context = {}

def get_conversation_context(session_id):
    """Get or initialize conversation context for a session"""
    if session_id not in conversation_context:
        conversation_context[session_id] = {
            'history': [],
            'last_intent': None,
            'pending_actions': [],
            'extracted_entities': {}
        }
    return conversation_context[session_id]

def update_conversation_context(session_id, user_message, bot_response, intent, entities=None):
    """Update the conversation context with the latest interaction"""
    context = get_conversation_context(session_id)
    
    # Add to history (keeping last 10 messages)
    context['history'].append({
        'user': user_message,
        'bot': bot_response,
        'intent': intent,
        'timestamp': datetime.now().isoformat()
    })
    context['history'] = context['history'][-10:]  # Keep only last 10 messages
    
    # Update last intent
    context['last_intent'] = intent
    
    # Update entities if provided
    if entities:
        if 'extracted_entities' not in context:
            context['extracted_entities'] = {}
        for entity_type, values in entities.items():
            if entity_type not in context['extracted_entities']:
                context['extracted_entities'][entity_type] = []
            for value in values:
                if value not in context['extracted_entities'][entity_type]:
                    context['extracted_entities'][entity_type].append(value)
    
    return context

# Track conversation state for availability checking
availability_state = {}

def check_room_availability(check_in, check_out, room_type=None):
    """Check room availability in the database
    
    Args:
        check_in (str): Check-in date in YYYY-MM-DD format
        check_out (str): Check-out date in YYYY-MM-DD format
        room_type (str, optional): Type of room to check. Defaults to None.
    
    Returns:
        int: Number of available rooms of the specified type
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # First, get the total number of rooms of the requested type
        room_count_query = """
        SELECT COUNT(*) as total_rooms
        FROM rooms r
        JOIN room_types rt ON r.room_type = rt.name
        WHERE rt.name = %s
        """
        
        # If no specific room type requested, get all room types
        if not room_type:
            room_count_query = """
            SELECT rt.name as room_type, COUNT(*) as total_rooms
            FROM rooms r
            JOIN room_types rt ON r.room_type = rt.name
            GROUP BY rt.name
            """
            cursor.execute(room_count_query)
            room_counts = cursor.fetchall()
            
            # If no room types found, return 0
            if not room_counts:
                cursor.close()
                conn.close()
                return 0
                
            # Get the minimum available rooms across all room types
            min_available = float('inf')
            for room in room_counts:
                available = _get_available_room_count(
                    cursor, room['room_type'], check_in, check_out, room['total_rooms']
                )
                min_available = min(min_available, available)
                
            cursor.close()
            conn.close()
            return min_available if min_available != float('inf') else 0
        
        # For specific room type
        cursor.execute(room_count_query, (room_type,))
        result = cursor.fetchone()
        total_rooms = result['total_rooms'] if result else 0
        
        if total_rooms == 0:
            cursor.close()
            conn.close()
            return 0
            
        # Get number of booked rooms for the date range
        available_rooms = _get_available_room_count(
            cursor, room_type, check_in, check_out, total_rooms
        )
        
        cursor.close()
        conn.close()
        return available_rooms
        
    except Exception as e:
        print(f"Error checking availability: {e}")
        return 0  # Return 0 available rooms in case of error

def _get_available_room_count(cursor, room_type, check_in, check_out, total_rooms):
    """Helper function to calculate available rooms for a specific room type"""
    query = """
    SELECT COUNT(DISTINCT r.id) as booked_rooms
    FROM rooms r
    JOIN bookings b ON r.room_type = b.room_type
    WHERE 
        r.room_type = %s
        AND b.status != 'cancelled'
        AND (
            (%s BETWEEN b.check_in AND DATE_SUB(b.check_out, INTERVAL 1 DAY))
            OR (%s BETWEEN b.check_in AND DATE_SUB(b.check_out, INTERVAL 1 DAY))
            OR (b.check_in BETWEEN %s AND %s)
            OR (DATE_SUB(b.check_out, INTERVAL 1 DAY) BETWEEN %s AND %s)
        )
    """
    
    cursor.execute(query, (
        room_type,
        check_in, check_out,
        check_in, check_out,
        check_in, check_out
    ))
    
    result = cursor.fetchone()
    booked_rooms = result['booked_rooms'] if result else 0
    return max(0, total_rooms - booked_rooms)

@app.route('/api/check-availability', methods=['POST'])
def check_availability_api():
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['check_in', 'check_out', 'room_type']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'available': False
            }), 400
        
        # Check room availability
        available_rooms = check_room_availability(
            data['check_in'],
            data['check_out'],
            data['room_type']
        )
        
        if available_rooms > 0:
            return jsonify({
                'available': True,
                'message': f'Great news! We have {available_rooms} {data["room_type"]} room(s) available for your selected dates.',
                'room_type': data['room_type'],
                'check_in': data['check_in'],
                'check_out': data['check_out']
            })
        else:
            return jsonify({
                'available': False,
                'message': 'Sorry, no rooms available for the selected dates and room type. Please try different dates or room type.',
                'room_type': data['room_type']
            })
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'available': False
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not user_message:
            return jsonify({'response': "I didn't catch that. Could you please repeat?"})
        
        # Get conversation context
        context = get_conversation_context(session_id)
        
        # Process the message with NLP
        nlp_result = chatbot.process_message(user_message, context)
        intent = nlp_result['intent']
        entities = nlp_result['entities']
        
        print("\n=== DEBUG: Processing Message ===")
        print(f"User message: {user_message}")
        print(f"Detected intent: {intent}")
        print(f"Confidence: {nlp_result.get('confidence', 'N/A')}")
        print(f"Entities: {entities}")
        print(f"Previous intent: {context.get('last_intent')}")
        print("=== End of Debug ===\n")
        
        # Check for specific intents based on keywords
        user_message_lower = user_message.lower()
        
        # Check for price-related queries
        rate_keywords = ['rate', 'price', 'cost', 'how much', 'how much does it cost']
        if any(keyword in user_message_lower for keyword in rate_keywords):
            print("[DEBUG] Message appears to be about room rates")
            intent = 'price_inquiry'
            
        # Check for availability-related queries
        availability_keywords = ['available', 'availability', 'book a room', 'rooms available', 'check availability', 'can i book']
        if any(keyword in user_message_lower for keyword in availability_keywords):
            print("[DEBUG] Message appears to be about room availability")
            # Only override if the current intent is not already a booking-related one
            if intent not in ['book_room', 'check_availability']:
                intent = 'check_availability'
        
        # Handle follow-up questions based on conversation context
        if intent == 'fallback' and context['history']:
            last_interaction = context['history'][-1]
            if 'room' in last_interaction.get('bot', '').lower() and 'available' in last_interaction.get('bot', '').lower():
                # If the bot asked about room type and got an unclear response
                room_type = extract_room_type(user_message)
                if room_type:
                    intent = 'check_availability'
                    if 'ROOM_TYPE' not in entities:
                        entities['ROOM_TYPE'] = []
                    entities['ROOM_TYPE'].append(room_type)
        
        # Initialize session state if it doesn't exist
        if session_id not in availability_state:
            availability_state[session_id] = {
                'checking_availability': False,
                'pending_booking': None
            }
        
        # Default response data
        response_data = {
            'response': '',
            'intent': intent,
            'entities': entities,
            'show_booking_form': False,
            'show_availability_check': False
        }
        
        # Special handling for specific intents that need custom UI elements or database operations
        if intent in ['book_room', 'check_availability']:
            # Show booking form for both booking and availability checks
            response_data['show_booking_form'] = True
            
            # If it's a follow-up booking, use a different response
            if intent == 'book_room' and ('another' in user_message.lower() or 'new' in user_message.lower() or 'one more' in user_message.lower()):
                response_data['response'] = get_response(intent, entities, user_message)
                return jsonify(response_data)
            
            # Get appropriate response for the intent
            response_data['response'] = get_response(intent, entities, user_message)
                
            # Extract booking details if available in the message
            details = extract_booking_details(entities, user_message)
            required_details = ['check_in', 'check_out', 'room_type']
            has_all_details = all(details.get(field) for field in required_details)
            
            # If all details are present, we can check availability directly
            if has_all_details and intent == 'check_availability':
                # This will be handled by the frontend's check availability button
                response_data['show_availability_check'] = True
            
            if has_all_details:
                conn = get_db_connection()
                cursor = conn.cursor(dictionary=True)
                guest_name = "Guest"
                
                try:
                    cursor.execute('''
                        INSERT INTO bookings (guest_name, check_in, check_out, room_type, status, guests)
                        VALUES (%s, %s, %s, %s, 'confirmed', %s)
                    ''', (
                        guest_name, 
                        details['check_in'], 
                        details['check_out'], 
                        details['room_type'],
                        details['guests']
                    ))
                    
                    booking_id = cursor.lastrowid
                    conn.commit()
                    
                    response_data['response'] = f"✅ Booking confirmed!\n\n" \
                        f"📅 Check-in: {details['check_in']}\n" \
                        f"🏨 Room Type: {details['room_type']}\n" \
                        f"👥 Guests: {details['guests']}\n" \
                        f"📝 Booking ID: {booking_id}\n\n" \
                        f"Thank you for choosing our hotel!"
                    
                except Exception as e:
                    conn.rollback()
                    response_data['response'] = f"Sorry, I couldn't process your booking. Error: {str(e)}"
                finally:
                    cursor.close()
                    conn.close()
                return jsonify(response_data)
            else:
                response_data['response'] = get_response(intent, entities, user_message)
                response_data['show_booking_form'] = True
                return jsonify(response_data)
        
        # For check_availability, show the booking form
        if intent == 'check_availability':
            response_data['response'] = get_response(intent, entities, user_message)
            response_data['show_booking_form'] = True
            response_data['show_availability_check'] = True
            response_data['intent'] = 'check_availability'  # Ensure intent is included in response
            return jsonify(response_data)
        
        # For price_inquiry, just return the response without showing any forms
        if intent in ['price_inquiry', 'pricing']:
            print(f"[DEBUG] Handling price inquiry - Original intent: {intent}")
            response = get_response('price_inquiry', entities, user_message)
            print(f"[DEBUG] Price inquiry response: {response}")
            response_data['response'] = response
            response_data['show_booking_form'] = False
            response_data['show_availability_check'] = False
            response_data['intent'] = 'price_inquiry'  # Ensure consistent intent name
            print(f"[DEBUG] Response data: {response_data}")
            return jsonify(response_data)
            
        # Define which intents should show the booking form
        booking_form_intents = {'book_room', 'check_availability'}
        
        # Get the response for the detected intent
        response = get_response(intent, entities, user_message)
        
        # Prepare response data
        response_data.update({
            'response': response,
            'show_booking_form': intent in booking_form_intents,
            'show_availability_check': intent == 'check_availability',
            'intent': intent
        })
        
        # Force show booking form for booking and availability intents
        if intent in ['book_room', 'check_availability']:
            response_data['show_booking_form'] = True
            if intent == 'check_availability':
                response_data['show_availability_check'] = True
        
        print(f"\n=== DEBUG: Handling {intent} intent ===")
        print(f"User message: {user_message}")
        print(f"Show booking form: {response_data['show_booking_form']}")
        print(f"Show availability check: {response_data['show_availability_check']}")
        print("=== End of intent handling ===\n")
        
        # Generate response based on intent and context
        if intent == 'check_availability' and not availability_state[session_id]['checking_availability']:
            # Start a new availability check
            response_data['response'] = get_response(intent, entities, user_message)
            response_data['show_booking_form'] = True
            response_data['show_availability_check'] = True
            availability_state[session_id]['checking_availability'] = True
            
            # If we have dates but no room type, ask for room type
            if 'DATE' in entities and 'ROOM_TYPE' not in entities:
                response_data['response'] = "What type of room would you like? We have Standard, Deluxe, and Suite."
                
        elif intent == 'book_room':
            # If we have all required info, proceed to booking
            required_info = ['check_in', 'check_out', 'room_type', 'guests']
            if all(info in entities for info in required_info):
                response_data['response'] = "Great! Let's book your room. Please provide your name and contact details."
                response_data['show_booking_form'] = True
                availability_state[session_id]['checking_availability'] = False
            else:
                # Ask for missing information
                missing = [info for info in required_info if info not in entities]
                if 'check_in' in missing or 'check_out' in missing:
                    response_data['response'] = "When would you like to stay? Please provide check-in and check-out dates."
                elif 'room_type' in missing:
                    response_data['response'] = "What type of room would you like? We have Standard, Deluxe, and Suite."
                elif 'guests' in missing:
                    response_data['response'] = "How many guests will be staying?"
        else:
            # For fallback intent, use the fallback response
            if intent == 'fallback':
                response_data['response'] = get_response('fallback', entities, user_message)
                response_data['show_booking_form'] = False
                response_data['show_availability_check'] = False
            # For all other intents, generate response based on context
            else:
                response_data['response'] = get_response(intent, entities, user_message)
                
                # Handle follow-up questions based on conversation context
                if availability_state[session_id]['checking_availability']:
                    if 'yes' in user_message.lower() and 'book' in user_message.lower():
                        intent = 'book_room'
                        response_data['response'] = "Great! Let's book your room. Please provide your name and contact details."
                        response_data['show_booking_form'] = True
                    elif 'no' in user_message.lower():
                        availability_state[session_id]['checking_availability'] = False
                        response_data['response'] = "No problem! Is there anything else I can help you with?"
        
        # Update conversation context
        update_conversation_context(
            session_id=session_id,
            user_message=user_message,
            bot_response=response_data['response'],
            intent=intent,
            entities=entities
        )
        
        # Add context to response for debugging
        if 'debug' in request.args:
            response_data['debug'] = {
                'intent': intent,
                'entities': entities,
                'last_intent': context.get('last_intent'),
                'conversation_length': len(context['history'])
            }
        
        print(f"[DEBUG] Sending response to frontend: {response_data}")
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'response': "I'm sorry, I encountered an error. Please try again.",
            'error': str(e)
        }), 500

@app.route('/api/bookings', methods=['POST'])
def create_booking():
    try:
        data = request.json
        required_fields = ['guest_name', 'check_in', 'check_out', 'room_type']
        
        # Validate required fields
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Validate dates
        from datetime import datetime
        try:
            check_in = datetime.strptime(data['check_in'], '%Y-%m-%d')
            check_out = datetime.strptime(data['check_out'], '%Y-%m-%d')
            
            if check_in >= check_out:
                return jsonify({'error': 'Check-out date must be after check-in date'}), 400
                
            if check_in < datetime.now():
                return jsonify({'error': 'Check-in date cannot be in the past'}), 400
                
        except ValueError as e:
            return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD'}), 400
        
        # Validate room type
        valid_room_types = ['Standard', 'Deluxe', 'Suite']
        if data['room_type'] not in valid_room_types:
            return jsonify({'error': f'Invalid room type. Must be one of: {", ".join(valid_room_types)}'}), 400
        
        # Check room availability using the same function as the availability check
        available_rooms = check_room_availability(
            data['check_in'],
            data['check_out'],
            data['room_type']
        )
        
        if available_rooms <= 0:
            return jsonify({
                'error': f'Sorry, the {data["room_type"]} room is not available for the selected dates. Please try different dates or room type.'
            }), 400
            
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get number of guests, default to 1 if not provided
        num_guests = int(data.get('guests', 1))
        
        # Create the booking
        cursor.execute('''
            INSERT INTO bookings (guest_name, check_in, check_out, room_type, guests, status)
            VALUES (%s, %s, %s, %s, %s, 'confirmed')
        ''', (data['guest_name'], check_in, check_out, data['room_type'], num_guests))
        
        booking_id = cursor.lastrowid
        conn.commit()
        
        # Calculate total price
        total_price = calculate_total_price(
            data['room_type'],
            check_in.strftime('%Y-%m-%d'),
            check_out.strftime('%Y-%m-%d')
        )
        
        return jsonify({
            'message': 'Booking created successfully',
            'booking_id': booking_id,
            'check_in': check_in.strftime('%Y-%m-%d'),
            'check_out': check_out.strftime('%Y-%m-%d'),
            'room_type': data['room_type'],
            'total_price': total_price,
            'guests': int(data.get('guests', 1))
        }), 201
    
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
            if 'cursor' in locals():
                cursor.close()
            conn.close()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    initialize_database()
    app.run(debug=True)
