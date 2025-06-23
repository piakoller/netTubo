import requests
import json
import os
from dotenv import load_dotenv
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_openrouter_api():
    """Test the OpenRouter API and print the response"""
    
    # Load environment variables from the correct path
    env_path = Path(__file__).parent / '.env'
    print(f"Loading .env from: {env_path}")
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(api_key)
    
    if not api_key:
        print("❌ ERROR: No API key found in .env file")
        return
        
    print(f"API Key loaded (first 4 chars): {api_key[:4]}...")
    print("Testing OpenRouter API...")
    print("-" * 50)
    
    try:
        # Updated headers according to OpenRouter API documentation
        headers = {
            "Authorization": api_key,  # Remove 'Bearer ' prefix
            "HTTP-Referer": "https://github.com/piakoller/netTubo",
            "X-Title": "NET Tumorboard Assistant",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "google/gemma-2-9b-it:free",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the meaning of life?"
                }
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        print("Request Headers:", {k: v if k != 'Authorization' else f"{v[:8]}..." for k, v in headers.items()})
        print("Request Data:", json.dumps(data, indent=2))
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("-" * 50)
        
        # Always print response text for debugging
        print("Raw Response:", response.text)
        print("-" * 50)
        
        if response.status_code == 200:
            response_data = response.json()
            print("✅ SUCCESS! API call successful")
            print("\nFull Response JSON:")
            print(json.dumps(response_data, indent=2))
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                print("\n" + "="*50)
                print("AI RESPONSE:")
                print("="*50)
                print(content)
                print("="*50)
            else:
                print("❌ No content found in response")
                
        else:
            print(f"❌ ERROR! Status code: {response.status_code}")
            print(f"Response text: {response.text}")
            
            try:
                error_data = response.json()
                print("Error details:")
                print(json.dumps(error_data, indent=2))
            except:
                print("Could not parse error response as JSON")
                
    except requests.exceptions.Timeout:
        print("❌ ERROR! Request timed out")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR! Connection error - check your internet connection")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR! Request exception: {e}")
        
    except Exception as e:
        print(f"❌ ERROR! Unexpected error: {e}")

if __name__ == "__main__":
    test_openrouter_api()