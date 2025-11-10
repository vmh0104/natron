"""
Example Usage Script
Demonstrates how to use Natron Transformer for predictions
"""

import pandas as pd
import requests
import json
from datetime import datetime, timedelta


def example_api_prediction():
    """Example: Using Flask API for predictions"""
    print("=" * 60)
    print("Example: Flask API Prediction")
    print("=" * 60)
    
    # Generate sample OHLCV data
    sample_candles = []
    base_price = 1.0
    for i in range(96):  # Need 96 candles
        candle = {
            "time": (datetime.now() - timedelta(minutes=96-i)).isoformat(),
            "open": base_price + (i * 0.001),
            "high": base_price + (i * 0.001) + 0.01,
            "low": base_price + (i * 0.001) - 0.01,
            "close": base_price + (i * 0.001) + 0.005,
            "volume": 1000 + (i * 10)
        }
        sample_candles.append(candle)
    
    # Make API request
    api_url = "http://localhost:5000/predict"
    payload = {"candles": sample_candles}
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Prediction received:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ API error: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server.")
        print("   Make sure the server is running: python api_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_socket_prediction():
    """Example: Using Socket server for predictions"""
    print("\n" + "=" * 60)
    print("Example: Socket Server Prediction")
    print("=" * 60)
    
    import socket
    
    # Generate sample data
    sample_candles = []
    base_price = 1.0
    for i in range(96):
        candle = {
            "time": (datetime.now() - timedelta(minutes=96-i)).isoformat(),
            "open": base_price + (i * 0.001),
            "high": base_price + (i * 0.001) + 0.01,
            "low": base_price + (i * 0.001) - 0.01,
            "close": base_price + (i * 0.001) + 0.005,
            "volume": 1000 + (i * 10)
        }
        sample_candles.append(candle)
    
    # Connect to socket server
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 8888))
        
        # Build request
        request = {
            "type": "predict",
            "candles": sample_candles
        }
        request_json = json.dumps(request) + '\n'
        
        # Send request
        sock.send(request_json.encode('utf-8'))
        
        # Receive response
        response = sock.recv(4096).decode('utf-8')
        result = json.loads(response.strip())
        
        print("\n✅ Prediction received:")
        print(json.dumps(result, indent=2))
        
        sock.close()
    
    except ConnectionRefusedError:
        print("❌ Could not connect to socket server.")
        print("   Make sure the server is running: python socket_server.py")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_training_workflow():
    """Example: Complete training workflow"""
    print("\n" + "=" * 60)
    print("Example: Training Workflow")
    print("=" * 60)
    
    print("""
    To train the Natron model, follow these steps:
    
    1. Prepare your data:
       - Ensure data_export.csv exists with columns: time, open, high, low, close, volume
       - Validate data: python validate_data.py
    
    2. Phase 1 - Pretraining:
       python train_pretrain.py
       (or: bash start_natron.sh train-pretrain)
    
    3. Phase 2 - Supervised Fine-tuning:
       python train_supervised.py
       (or: bash start_natron.sh train-supervised)
    
    4. Phase 3 - Reinforcement Learning (optional):
       python train_rl.py
       (or: bash start_natron.sh train-rl)
    
    5. Or train all phases:
       python train_all.py
       (or: bash start_natron.sh train-all)
    
    After training, models will be saved in ./models/
    """)


def main():
    print("\n🧠 Natron Transformer - Example Usage\n")
    
    # Show training workflow
    example_training_workflow()
    
    # Show API usage
    example_api_prediction()
    
    # Show socket usage
    example_socket_prediction()
    
    print("\n" + "=" * 60)
    print("For more information, see README.md")
    print("=" * 60)


if __name__ == '__main__':
    main()
