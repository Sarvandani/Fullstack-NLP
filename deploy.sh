#!/bin/bash

# Deployment Package Creator
# Creates a clean package for uploading to hosting services

echo "📦 Creating deployment package..."

# Clean up
echo "🧹 Cleaning up..."
rm -rf models/__pycache__ backend/__pycache__ frontend/__pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
rm -f *.log backend.log frontend.log

# Create deployment directory
DEPLOY_DIR="deploy_package"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy necessary files
echo "📋 Copying files..."
cp -r backend/ $DEPLOY_DIR/
cp -r frontend/ $DEPLOY_DIR/
cp setup_models.py $DEPLOY_DIR/ 2>/dev/null || true
cp Dockerfile $DEPLOY_DIR/ 2>/dev/null || true
cp .dockerignore $DEPLOY_DIR/ 2>/dev/null || true
cp README.md $DEPLOY_DIR/ 2>/dev/null || true

# Remove unnecessary files
echo "🗑️  Removing unnecessary files..."
rm -rf $DEPLOY_DIR/models
rm -f $DEPLOY_DIR/**/*.log
rm -f $DEPLOY_DIR/**/__pycache__

# Create archive
echo "📦 Creating archive..."
tar -czf deploy.tar.gz $DEPLOY_DIR/ 2>/dev/null || zip -r deploy.zip $DEPLOY_DIR/

echo ""
echo "✅ Deployment package created!"
echo "📁 Directory: $DEPLOY_DIR/"
if [ -f "deploy.tar.gz" ]; then
    echo "📦 Archive: deploy.tar.gz"
    echo "   Size: $(du -h deploy.tar.gz | cut -f1)"
fi
if [ -f "deploy.zip" ]; then
    echo "📦 Archive: deploy.zip"
    echo "   Size: $(du -h deploy.zip | cut -f1)"
fi
echo ""
echo "🚀 Next steps:"
echo "   1. Upload $DEPLOY_DIR/ or deploy.tar.gz to your hosting service"
echo "   2. Models will download automatically on first run"
echo "   3. Update frontend API URL to your backend URL"
echo ""

