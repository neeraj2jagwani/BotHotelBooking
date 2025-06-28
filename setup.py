import os
import subprocess
import sys

def install_requirements():
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_spacy_model():
    print("Downloading spaCy English model...")
    try:
        import spacy
        spacy.cli.download("en_core_web_sm")
        print("Successfully downloaded spaCy English model.")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
        sys.exit(1)

def train_nlp_model():
    print("Training the NLP model...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error training the model: {e}")
        sys.exit(1)

def main():
    # Create necessary directories
    os.makedirs("models/nlp_model", exist_ok=True)
    os.makedirs("training", exist_ok=True)
    
    # Install requirements
    install_requirements()
    
    # Download spaCy model
    download_spacy_model()
    
    # Train the NLP model
    train_nlp_model()
    
    print("\nSetup completed successfully!")
    print("You can now start the application with: python app.py")

if __name__ == "__main__":
    main()
