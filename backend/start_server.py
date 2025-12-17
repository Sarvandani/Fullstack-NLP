#!/usr/bin/env python3
"""
Startup script for the NLP backend server
Handles model loading and port detection
"""

import sys
import os
import socket
import uvicorn

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_available_port(start_port=5000, max_port=5100):
    """Find an available port starting from start_port"""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise Exception(f"No available ports found between {start_port} and {max_port}")

def main():
    print("=" * 60)
    print("🚀 Starting NLP Backend Server")
    print("=" * 60)
    
    # Find available port
    port = find_available_port(5000, 5100)
    print(f"\n📡 Starting server on port {port}")
    print(f"🌐 Frontend should connect to: http://localhost:{port}")
    print(f"\n⏳ Loading NLP models (this may take a minute on first run)...")
    print("=" * 60)
    
    try:
        # Import here so we can catch errors
        from main import app
        print("\n✅ Server starting...")
        print(f"📊 API available at: http://localhost:{port}")
        print(f"❤️  Health check: http://localhost:{port}/health")
        print("\n" + "=" * 60)
        
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

