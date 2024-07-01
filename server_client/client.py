import requests

api_url = "http://localhost:8000/query"

query = input("Input Prompt: ")
chat_history = []

while query.lower() != "exit":
    response = requests.post(api_url, json={"question": query, "chat_history": chat_history})
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data['answer']}")
        print(f"Response time: {data['response_time']} seconds.")
    else:
        print("Error:", response.status_code, response.text)
    
    query = input("Input Prompt: ")
