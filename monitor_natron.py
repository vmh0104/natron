"""
Natron Monitoring Script
Monitor model performance and server health.
"""

import requests
import json
import time
from datetime import datetime
import sys


def check_server_health(flask_port=5000):
    """Check Flask server health"""
    try:
        response = requests.get(f'http://localhost:{flask_port}/health', timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server Status: {data.get('status', 'unknown')}")
            print(f"  Device: {data.get('device', 'unknown')}")
            return True
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Is server_natron.py running?")
        return False
    except Exception as e:
        print(f"✗ Error checking server: {e}")
        return False


def test_prediction(flask_port=5000):
    """Test prediction endpoint with sample data"""
    # Generate sample candles (96 candles)
    sample_candles = []
    base_price = 1.1000
    
    for i in range(96):
        candle = {
            "time": int(time.time()) - (96 - i) * 3600,
            "open": base_price + i * 0.0001,
            "high": base_price + i * 0.0001 + 0.0010,
            "low": base_price + i * 0.0001 - 0.0005,
            "close": base_price + i * 0.0001 + 0.0005,
            "volume": 1000 + i * 10
        }
        sample_candles.append(candle)
    
    try:
        response = requests.post(
            f'http://localhost:{flask_port}/predict',
            json={'candles': sample_candles},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print("\n📊 Prediction Results:")
            print(f"  Buy Probability: {data.get('buy_prob', 0):.2%}")
            print(f"  Sell Probability: {data.get('sell_prob', 0):.2%}")
            print(f"  Direction Up: {data.get('direction_up', 0):.2%}")
            print(f"  Market Regime: {data.get('regime', 'UNKNOWN')}")
            print(f"  Confidence: {data.get('confidence', 0):.2%}")
            return True
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error testing prediction: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Natron Server')
    parser.add_argument('--flask-port', type=int, default=5000, help='Flask port')
    parser.add_argument('--interval', type=int, default=10, help='Check interval (seconds)')
    parser.add_argument('--continuous', action='store_true', help='Continuous monitoring')
    args = parser.parse_args()
    
    print("=" * 50)
    print("Natron Transformer Monitor")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Health check
    print("1. Health Check:")
    health_ok = check_server_health(args.flask_port)
    
    if not health_ok:
        print("\n❌ Server is not healthy. Exiting.")
        sys.exit(1)
    
    # Test prediction
    print("\n2. Prediction Test:")
    prediction_ok = test_prediction(args.flask_port)
    
    if not prediction_ok:
        print("\n❌ Prediction test failed. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ All checks passed!")
    print("=" * 50)
    
    # Continuous monitoring
    if args.continuous:
        print(f"\n🔄 Starting continuous monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                time.sleep(args.interval)
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking...")
                check_server_health(args.flask_port)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")


if __name__ == '__main__':
    main()
