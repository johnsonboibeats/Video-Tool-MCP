#!/usr/bin/env python3
"""
Test server startup
"""

import subprocess
import time
import signal
import sys

def test_server_startup():
    """Test if the server can start without errors"""
    print("🚀 Testing server startup...")
    
    try:
        # Start server in background
        process = subprocess.Popen(
            [sys.executable, "server.py", "--transport", "http", "--port", "8001"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a few seconds for startup
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Server started successfully")
            
            # Terminate the process
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ Server terminated cleanly")
            except subprocess.TimeoutExpired:
                process.kill()
                print("⚠️ Server had to be force killed")
            
            return True
        else:
            # Get error output
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start")
            print(f"Exit code: {process.returncode}")
            if stderr:
                print(f"Error output: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

if __name__ == "__main__":
    success = test_server_startup()
    sys.exit(0 if success else 1) 