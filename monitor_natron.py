"""
Natron Monitoring Script - Monitor model performance and system health
"""

import requests
import json
import time
import psutil
import torch
from datetime import datetime


def check_api_health(url: str = "http://localhost:5000/health") -> dict:
    """Check API server health."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "unhealthy", "code": response.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_system_stats() -> dict:
    """Get system statistics."""
    stats = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        "timestamp": datetime.now().isoformat()
    }
    
    # GPU stats if available
    if torch.cuda.is_available():
        stats["gpu_available"] = True
        stats["gpu_count"] = torch.cuda.device_count()
        stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
        stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
    else:
        stats["gpu_available"] = False
    
    return stats


def test_prediction(url: str = "http://localhost:5000/predict") -> dict:
    """Test prediction endpoint with dummy data."""
    # Generate dummy candles (96 candles)
    candles = []
    base_price = 100.0
    
    for i in range(96):
        candles.append({
            "time": int(time.time()) + i * 900,  # 15-minute intervals
            "open": base_price + i * 0.01,
            "high": base_price + i * 0.01 + 0.5,
            "low": base_price + i * 0.01 - 0.5,
            "close": base_price + i * 0.01 + 0.2,
            "volume": 1000 + i * 10
        })
    
    request_data = {"candles": candles}
    
    try:
        start_time = time.time()
        response = requests.post(url, json=request_data, timeout=10)
        latency = (time.time() - start_time) * 1000  # ms
        
        if response.status_code == 200:
            return {
                "status": "success",
                "latency_ms": round(latency, 2),
                "prediction": response.json()
            }
        else:
            return {
                "status": "error",
                "code": response.status_code,
                "message": response.text
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    print("=" * 60)
    print("Natron Monitoring Dashboard")
    print("=" * 60)
    print()
    
    # System stats
    print("📊 System Statistics:")
    sys_stats = get_system_stats()
    print(f"   CPU Usage: {sys_stats['cpu_percent']:.1f}%")
    print(f"   Memory Usage: {sys_stats['memory_percent']:.1f}%")
    print(f"   Memory Available: {sys_stats['memory_available_gb']:.2f} GB")
    
    if sys_stats.get('gpu_available'):
        print(f"   GPU Memory Allocated: {sys_stats['gpu_memory_allocated_gb']:.2f} GB")
        print(f"   GPU Memory Reserved: {sys_stats['gpu_memory_reserved_gb']:.2f} GB")
    
    print()
    
    # API health
    print("🏥 API Health Check:")
    health = check_api_health()
    print(f"   Status: {health['status']}")
    if 'data' in health:
        print(f"   Model Loaded: {health['data'].get('model_loaded', False)}")
    
    print()
    
    # Test prediction
    print("🧪 Testing Prediction Endpoint:")
    prediction_test = test_prediction()
    print(f"   Status: {prediction_test['status']}")
    if prediction_test['status'] == 'success':
        print(f"   Latency: {prediction_test['latency_ms']} ms")
        pred = prediction_test['prediction']
        print(f"   Buy Prob: {pred.get('buy_prob', 'N/A')}")
        print(f"   Sell Prob: {pred.get('sell_prob', 'N/A')}")
        print(f"   Regime: {pred.get('regime', 'N/A')}")
        print(f"   Confidence: {pred.get('confidence', 'N/A')}")
    
    print()
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Monitoring')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval (seconds)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    
    args = parser.parse_args()
    
    if args.once:
        main()
    else:
        print(f"Monitoring every {args.interval} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                main()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
