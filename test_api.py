import requests

test_data = {"text": "Put any sample news story here"}
response = requests.post("http://127.0.0.1:5000/predict", json=test_data)
print(response.json())