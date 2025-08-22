import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# This script tests your Google API key directly.
# Make sure you have run 'pip install google-generativeai'
# and set your GOOGLE_API_KEY environment variable.

try:
    # 1. Load the API key from environment variables
    api_key = os.getenv("GOOGLE_API")
    if not api_key:
        raise ValueError("ERROR: GOOGLE_API_KEY environment variable not set.")
    
    print("✅ Found GOOGLE_API_KEY.")
    
    # 2. Configure the library with your key
    genai.configure(api_key=api_key)
    print("✅ Library configured successfully.")

    # 3. Initialize the model
    # We use 'gemini-pro' here as it's the most basic text model, perfect for a simple test.
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    print("✅ Model 'gemini-pro' initialized.")

    # 4. Send a simple prompt to the API
    print("\n🚀 Sending a test prompt to the Gemini API...")
    prompt = "In one short sentence, what is a financial audit?"
    response = model.generate_content(prompt)
    
    # 5. Print the response
    print("\n🎉 SUCCESS! API responded with:")
    print("---------------------------------")
    print(response.text)
    print("---------------------------------")
    print("\nYour API key is working correctly!")

except Exception as e:
    print(f"\n❌ TEST FAILED.")
    print(f"An error occurred: {e}")