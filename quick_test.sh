#!/bin/bash
# Quick test commands for Enhanced DCGAN Research Package

echo "🚀 Enhanced DCGAN Research - Quick Local Testing"
echo "================================================"

# 1. Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
echo "✅ Cleaned"

# 2. Install build tools
echo "📦 Installing/upgrading build tools..."
pip install --upgrade build twine

# 3. Build the package
echo "🔨 Building package..."
python -m build

# 4. Check package quality
echo "🔍 Checking package quality..."
python -m twine check dist/*

# 5. List built files
echo "📁 Built files:"
ls -lh dist/

# 6. Test installation in development mode
echo "🧪 Testing development installation..."
pip install -e .

# 7. Quick import test
echo "🐍 Testing Python import..."
python -c "
import enhanced_dcgan_research as edr
print('✅ Package imported successfully')
print('📊 Package info:', edr.get_info())
print('📦 Available functions:', [attr for attr in dir(edr) if not attr.startswith('_')])
"

# 8. Test CLI command (if available)
echo "🖥️  Testing CLI command..."
enhanced-dcgan --help || echo "⚠️  CLI command not available (expected if not installed globally)"

echo ""
echo "🎉 Quick testing completed!"
echo "📝 Next steps:"
echo "   - If all tests passed: python build_and_upload.py"
echo "   - For detailed testing: python local_build_test.py"