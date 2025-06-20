import requests
import json

def test_openrouter_api():
    """Test the OpenRouter API and print the response"""
    
    print("Testing OpenRouter API...")
    print("-" * 50)
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "HTTP-Referer": "https://yoursite.com",  # Updated with actual URL
                "X-Title": "NET Tumorboard Assistant",     # Updated with actual name
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": "google/gemma-2-9b-it:free",  # Updated to a valid free model
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the meaning of life?"
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }),
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("-" * 50)
        
        if response.status_code == 200:
            response_data = response.json()
            print("✅ SUCCESS! API call successful")
            print("\nFull Response JSON:")
            print(json.dumps(response_data, indent=2))
            
            # Extract and print just the content
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
            
            # Try to parse error JSON
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