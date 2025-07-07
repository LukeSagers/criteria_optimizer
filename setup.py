#!/usr/bin/env python3
"""
Setup script for Clinical Trial Criteria Optimizer
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def check_streamlit():
    """Check if Streamlit is available."""
    try:
        import streamlit
        print("✅ Streamlit is available")
        return True
    except ImportError:
        print("❌ Streamlit not found")
        return False

def run_app():
    """Run the Streamlit application."""
    print("Starting Clinical Trial Criteria Optimizer...")
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main setup function."""
    print("🏥 Clinical Trial Criteria Optimizer Setup")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        print("Please make sure you're in the correct directory.")
        return
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("❌ app.py not found!")
        print("Please make sure you're in the correct directory.")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check if streamlit is available
    if not check_streamlit():
        print("Please install the requirements manually:")
        print("pip install -r requirements.txt")
        return
    
    print("\n🎉 Setup complete!")
    print("=" * 50)
    
    # Ask user if they want to run the app
    response = input("Would you like to start the application now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        run_app()
    else:
        print("\nTo run the application later, use:")
        print("streamlit run app.py")

if __name__ == "__main__":
    main() 