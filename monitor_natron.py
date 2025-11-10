#!/usr/bin/env python3
"""
Natron Monitoring Script
Monitor training progress, server status, and model performance

Usage:
    python monitor_natron.py --logs logs/
    python monitor_natron.py --server http://localhost:8888
"""

import argparse
import requests
import time
import os
import sys
from datetime import datetime
import json


def check_server_health(base_url: str):
    """Check if Natron server is healthy"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        return False, None
    except Exception as e:
        return False, str(e)


def get_model_info(base_url: str):
    """Get model information from server"""
    try:
        response = requests.get(f"{base_url}/model_info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def monitor_server(base_url: str, interval: int = 10):
    """Continuously monitor server status"""
    print("="*60)
    print("🔍 Natron Server Monitor")
    print("="*60)
    print(f"Server: {base_url}")
    print(f"Check interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check health
            is_healthy, health_data = check_server_health(base_url)
            
            if is_healthy:
                status = "🟢 HEALTHY"
                device = health_data.get('device', 'unknown')
                model_loaded = health_data.get('model_loaded', False)
                
                print(f"[{timestamp}] {status} | Device: {device} | Model: {'✅' if model_loaded else '❌'}")
            else:
                status = "🔴 OFFLINE"
                print(f"[{timestamp}] {status} | Error: {health_data}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped")


def tail_predictions_log(log_path: str, n_lines: int = 20):
    """Display recent predictions from log file"""
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return
    
    print("="*60)
    print(f"📊 Recent Predictions (last {n_lines})")
    print("="*60)
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        recent = lines[-n_lines:]
        
        for line in recent:
            try:
                pred = json.loads(line.strip())
                timestamp = datetime.fromtimestamp(pred['timestamp']).strftime("%H:%M:%S")
                
                buy = pred['buy_prob'] * 100
                sell = pred['sell_prob'] * 100
                regime = pred['regime']
                conf = pred['confidence'] * 100
                
                # Signal indicator
                signal = "  "
                if buy >= 65:
                    signal = "🟢"
                elif sell >= 65:
                    signal = "🔴"
                else:
                    signal = "⚪"
                
                print(f"[{timestamp}] {signal} Buy:{buy:5.1f}% Sell:{sell:5.1f}% | {regime:12s} | Conf:{conf:5.1f}%")
                
            except Exception as e:
                continue


def display_training_summary(checkpoint_dir: str):
    """Display summary of training checkpoints"""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    print("="*60)
    print("📁 Training Checkpoints")
    print("="*60)
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoints:
        print("No checkpoints found")
        return
    
    for ckpt in sorted(checkpoints):
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(ckpt_path))
        
        print(f"📦 {ckpt:30s} | {size_mb:6.2f} MB | {mod_time.strftime('%Y-%m-%d %H:%M')}")


def main():
    parser = argparse.ArgumentParser(description='Natron Monitoring Tool')
    parser.add_argument('--server', type=str, help='Server URL (e.g., http://localhost:8888)')
    parser.add_argument('--interval', type=int, default=10, help='Check interval in seconds')
    parser.add_argument('--logs', type=str, help='Path to logs directory')
    parser.add_argument('--predictions', action='store_true', help='Show recent predictions')
    parser.add_argument('--checkpoints', type=str, help='Path to checkpoint directory')
    
    args = parser.parse_args()
    
    if args.server:
        # Monitor server
        monitor_server(args.server, args.interval)
        
    elif args.predictions:
        # Show predictions
        log_path = 'logs/predictions.log'
        if args.logs:
            log_path = os.path.join(args.logs, 'predictions.log')
        tail_predictions_log(log_path, n_lines=30)
        
    elif args.checkpoints:
        # Show checkpoints
        display_training_summary(args.checkpoints)
        
    else:
        # Default: show all info
        print("\n🧠 Natron System Status\n")
        
        # Server status
        server_url = 'http://localhost:8888'
        is_healthy, health_data = check_server_health(server_url)
        
        print("🌐 Server Status:")
        if is_healthy:
            print(f"   ✅ Server is running at {server_url}")
            model_info = get_model_info(server_url)
            if model_info:
                print(f"   📊 Model: {model_info.get('model', 'Unknown')}")
                print(f"   🔢 Parameters: {model_info.get('parameters', 0):,}")
        else:
            print(f"   ❌ Server is offline")
        
        # Checkpoints
        print("\n📁 Checkpoints:")
        for phase in ['pretrain', 'supervised']:
            ckpt_dir = f'models/{phase}'
            if os.path.exists(ckpt_dir):
                ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
                print(f"   {phase.capitalize()}: {len(ckpts)} checkpoint(s)")
        
        # Predictions log
        log_path = 'logs/predictions.log'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                lines = f.readlines()
                print(f"\n📊 Predictions: {len(lines)} total")
                if lines:
                    tail_predictions_log(log_path, n_lines=10)
        
        print("\n" + "="*60)
        print("💡 Usage:")
        print("   python monitor_natron.py --server http://localhost:8888")
        print("   python monitor_natron.py --predictions")
        print("   python monitor_natron.py --checkpoints models/supervised")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
