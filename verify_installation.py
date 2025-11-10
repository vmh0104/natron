#!/usr/bin/env python3
"""
Natron Installation Verification Script
Checks all components and dependencies

Usage:
    python verify_installation.py
"""

import sys
import os
from pathlib import Path


def print_section(title):
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version >= (3, 10):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_dependencies():
    """Check Python dependencies"""
    required = {
        'torch': 'PyTorch',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'flask': 'Flask',
        'yaml': 'PyYAML'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Not installed")
            missing.append(name)
    
    return len(missing) == 0, missing


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Devices: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠️  CUDA not available (will use CPU)")
            return True  # Not fatal
    except:
        print("❌ Could not check CUDA")
        return False


def check_files():
    """Check if all required files exist"""
    required_files = [
        'config.yaml',
        'requirements.txt',
        'README.md',
        'train_natron.py',
        'server_natron.py',
        'inference.py',
        'natron_ea.mq5',
        'src/__init__.py',
        'src/feature_engine.py',
        'src/label_generator.py',
        'src/dataset_loader.py',
        'src/model_natron.py',
        'src/losses.py',
        'src/train_pretrain.py',
        'src/train_supervised.py'
    ]
    
    missing = []
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            missing.append(file)
    
    return len(missing) == 0, missing


def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        'src',
        'data',
        'models',
        'models/pretrain',
        'models/supervised',
        'logs'
    ]
    
    missing = []
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists() and path.is_dir():
            print(f"✅ {dir_name}/")
        else:
            print(f"⚠️  {dir_name}/ - Creating...")
            path.mkdir(parents=True, exist_ok=True)
    
    return True


def check_config():
    """Check configuration file"""
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['data', 'model', 'pretrain', 'supervised', 'api']
        for key in required_keys:
            if key in config:
                print(f"✅ Config section: {key}")
            else:
                print(f"❌ Config section missing: {key}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False


def test_imports():
    """Test importing core modules"""
    sys.path.append('src')
    
    modules = [
        'feature_engine',
        'label_generator',
        'dataset_loader',
        'model_natron',
        'losses'
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"✅ src.{module}")
        except Exception as e:
            print(f"❌ src.{module} - {str(e)[:50]}")
            failed.append(module)
    
    return len(failed) == 0, failed


def main():
    """Run all checks"""
    print("="*60)
    print("🧠 NATRON TRANSFORMER V2 - INSTALLATION VERIFICATION")
    print("="*60)
    
    results = {}
    
    # Python version
    print_section("Python Version")
    results['python'] = check_python_version()
    
    # Dependencies
    print_section("Python Dependencies")
    results['deps'], missing_deps = check_dependencies()
    
    # CUDA
    print_section("CUDA Support")
    results['cuda'] = check_cuda()
    
    # Files
    print_section("Project Files")
    results['files'], missing_files = check_files()
    
    # Directories
    print_section("Project Directories")
    results['dirs'] = check_directories()
    
    # Configuration
    print_section("Configuration")
    results['config'] = check_config()
    
    # Module imports
    print_section("Module Imports")
    results['imports'], failed_imports = test_imports()
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED!")
        print("\n🎉 Natron is ready to use!")
        print("\n📚 Next steps:")
        print("   1. Generate sample data: python generate_sample_data.py")
        print("   2. Train model: python train_natron.py")
        print("   3. Test inference: python inference.py --csv data/data_export.csv")
        print("   4. Start server: ./start_natron.sh")
        print("\n📖 Read QUICKSTART.md for detailed instructions")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\n❌ Issues found:")
        
        if not results['python']:
            print("   - Python version < 3.10")
        
        if not results['deps']:
            print(f"   - Missing dependencies: {', '.join(missing_deps)}")
            print("     Fix: pip install -r requirements.txt")
        
        if not results['files']:
            print(f"   - Missing files: {', '.join(missing_files)}")
        
        if not results['imports']:
            print(f"   - Import errors in: {', '.join(failed_imports)}")
        
        print("\n💡 Fix the issues above and run this script again")
    
    print("\n" + "="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
