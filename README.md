# AI-Powered Hotel Booking Chatbot

An intelligent hotel booking chatbot with natural language understanding, built with spaCy for NLP, Flask for the backend, and a modern web interface.

## Features

- **Natural Language Understanding**: Uses spaCy for intent recognition and entity extraction
- **Interactive Chat Interface**: Clean, responsive UI with message history
- **Room Booking System**: Check availability and book rooms with natural language
- **Contextual Responses**: Maintains conversation context for better user experience
- **Database Integration**: Stores bookings and user interactions in MySQL
- **Machine Learning**: Trained model for understanding various booking-related queries

## Quick Start

### Prerequisites

- Python 3.7+
- MySQL Server
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/neeraj2jagwani/BotHotelBooking.git
   cd hotel-booking-chatbot
   ```

2. **Run the setup script**
   This will install all dependencies and train the NLP model.
   ```bash
   python setup.py
   ```

3. **Configure the database**
   - Make sure MySQL is running
   - Update the database configuration in `app.py` if needed:
     ```python
     db_config = {
         'host': 'localhost',
         'user': 'your_username',
         'password': 'your_password',
         'database': 'hotel_booking_db'
     }
     ```

4. **Start the application**
   ```bash
   python app.py
   ```

5. **Access the chatbot**
   Open your browser and go to `http://localhost:5000`

## Training the Model

The chatbot uses a machine learning model for natural language understanding. The model is trained on the data in `training/train_intents.json`.

To retrain the model:
```bash
python train_model.py
```

## Project Structure

```
hotel-booking-chatbot/
├── app.py                # Main Flask application
├── nlp_model.py          # NLP model implementation
├── train_model.py        # Script to train the NLP model
├── training/
│   └── train_intents.json  # Training data for the chatbot
├── models/               # Trained model files
├── static/               # Static files (CSS, JS)
└── templates/           
    └── index.html        # Chatbot interface
```

## Example Queries

Try these example queries in the chat:

- "Hello"
- "What amenities does hotel have?"
- "Do you have wifi?"
- "Is there a swimming pool?"
- "What is check in and out time?"
- "What is the price of a room?"
- "I want to check the availability"
- "I want to cancel my booking"
- "I want to book a room"
- "<Insert random text to test robustness>"
- "Thank you"

## Customization

### Adding New Intents

1. Edit `training/train_intents.json`
2. Add a new intent with patterns and responses
3. Retrain the model: `python train_model.py`

### Modifying Responses

Edit the responses in `training/train_intents.json` and restart the application.

