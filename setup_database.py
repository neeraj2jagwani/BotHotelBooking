import mysql.connector
from datetime import datetime, timedelta
import random

def create_database():
    # Connect to MySQL server (without specifying a database)
    conn = mysql.connector.connect(
        host='localhost',
        user='root',  # Replace with your MySQL username
        password='root'   # Replace with your MySQL password
    )
    
    cursor = conn.cursor()
    
    try:
        # Create database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS hotel_booking_db")
        print("Database 'hotel_booking_db' created successfully or already exists.")
        
        # Switch to the database
        cursor.execute("USE hotel_booking_db")
        
        # Create bookings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bookings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                guest_name VARCHAR(100) NOT NULL,
                check_in DATE NOT NULL,
                check_out DATE NOT NULL,
                room_type VARCHAR(50) NOT NULL,
                guests INT DEFAULT 1,
                status VARCHAR(20) DEFAULT 'confirmed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        ''')
        
        # Create rooms table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rooms (
                id INT AUTO_INCREMENT PRIMARY KEY,
                room_number VARCHAR(10) NOT NULL UNIQUE,
                room_type VARCHAR(50) NOT NULL,
                price_per_night DECIMAL(10, 2) NOT NULL,
                max_guests INT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create room_types table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS room_types (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(50) NOT NULL UNIQUE,
                description TEXT,
                base_price DECIMAL(10, 2) NOT NULL,
                max_occupancy INT NOT NULL,
                amenities TEXT
            )
        ''')
        
        # Insert sample room types if they don't exist
        room_types = [
            ('Standard', 'Comfortable room with basic amenities', 99.99, 2, 'Free WiFi, TV, Air Conditioning'),
            ('Deluxe', 'Spacious room with premium amenities', 149.99, 3, 'Free WiFi, TV, Air Conditioning, Mini Bar, City View'),
            ('Suite', 'Luxury suite with separate living area', 249.99, 4, 'Free WiFi, TV, Air Conditioning, Mini Bar, Balcony, City View, Room Service')
        ]
        
        for room_type in room_types:
            cursor.execute('''
                INSERT IGNORE INTO room_types (name, description, base_price, max_occupancy, amenities)
                VALUES (%s, %s, %s, %s, %s)
            ''', room_type)
        
        # Insert sample rooms if they don't exist
        rooms = []
        room_numbers = set()
        
        # Generate 30 rooms (10 of each type)
        for i in range(1, 31):
            room_type = 'Standard' if i <= 10 else 'Deluxe' if i <= 20 else 'Suite'
            room_number = f"{room_type[0]}{i:02d}"
            price = 99.99 if room_type == 'Standard' else 149.99 if room_type == 'Deluxe' else 249.99
            max_guests = 2 if room_type == 'Standard' else 3 if room_type == 'Deluxe' else 4
            rooms.append((room_number, room_type, price, max_guests))
        
        for room in rooms:
            cursor.execute('''
                INSERT IGNORE INTO rooms (room_number, room_type, price_per_night, max_guests)
                VALUES (%s, %s, %s, %s)
            ''', room)
        
        # Create a few sample bookings
        today = datetime.now().date()
        sample_bookings = [
            ('John Doe', today + timedelta(days=5), today + timedelta(days=7), 'Deluxe', 2),
            ('Jane Smith', today + timedelta(days=2), today + timedelta(days=4), 'Standard', 1),
            ('Bob Johnson', today - timedelta(days=3), today + timedelta(days=2), 'Suite', 3)
        ]
        
        for booking in sample_bookings:
            cursor.execute('''
                INSERT INTO bookings (guest_name, check_in, check_out, room_type, guests)
                SELECT %s, %s, %s, %s, %s
                WHERE NOT EXISTS (
                    SELECT 1 FROM bookings 
                    WHERE guest_name = %s 
                    AND check_in = %s 
                    AND room_type = %s
                )
            ''', (*booking, booking[0], booking[1], booking[3]))
        
        conn.commit()
        print("Database tables created and sample data inserted successfully!")
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    print("Setting up the hotel booking database...")
    create_database()
    print("Database setup completed!")
