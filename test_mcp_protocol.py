#!/usr/bin/env python3
"""
Test MCP protocol properly
"""

import asyncio
import httpx
import json
import subprocess
import time

async def test_mcp_protocol():
    """Test MCP protocol properly"""
    print("🔧 Testing MCP Protocol...")
    
    async with httpx.AsyncClient() as client:
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            # Step 1: Initialize
            print("   Step 1: Initializing...")
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "Claude",
                        "version": "1.0"
                    }
                }
            }
            
            response = await client.post("http://localhost:8000/mcp/", json=init_request, headers=headers)
            print(f"   Init status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    init_data = response.json()
                    print("   ✅ Initialization successful")
                    print(f"   Server: {init_data.get('result', {}).get('serverInfo', {}).get('name')}")
                except json.JSONDecodeError:
                    print("   ⚠️ Response not JSON (might be SSE stream)")
                    print(f"   Response: {response.text[:200]}...")
            
            # Step 2: List tools
            print("   Step 2: Listing tools...")
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            response = await client.post("http://localhost:8000/mcp/", json=tools_request, headers=headers)
            print(f"   Tools status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    tools_data = response.json()
                    tools = tools_data.get('result', {}).get('tools', [])
                    print(f"   ✅ Found {len(tools)} tools:")
                    for tool in tools:
                        print(f"     - {tool.get('name')}: {tool.get('description', 'No description')}")
                    return True
                except json.JSONDecodeError:
                    print("   ⚠️ Tools response not JSON")
                    print(f"   Response: {response.text[:200]}...")
            else:
                print(f"   ❌ Tools request failed: {response.text}")
            
            return False
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

async def test_claude_desktop_compatibility():
    """Test if server is compatible with Claude Desktop"""
    print("\n🤖 Testing Claude Desktop Compatibility...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test health endpoint
            response = await client.get("http://localhost:8000/health")
            if response.status_code == 200:
                print("   ✅ Health endpoint working")
            else:
                print(f"   ❌ Health endpoint failed: {response.status_code}")
                return False
            
            # Test OAuth discovery
            response = await client.get("http://localhost:8000/.well-known/oauth-authorization-server")
            if response.status_code == 200:
                print("   ✅ OAuth discovery working")
            else:
                print(f"   ❌ OAuth discovery failed: {response.status_code}")
                return False
            
            # Test root endpoint
            response = await client.get("http://localhost:8000/")
            if response.status_code == 200:
                print("   ✅ Root endpoint working")
            else:
                print(f"   ❌ Root endpoint failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

async def main():
    """Run MCP protocol tests"""
    print("🔍 MCP Protocol Testing")
    print("=" * 40)
    
    # Start server
    print("🚀 Starting server...")
    server_process = subprocess.Popen(
        ["python3", "server.py", "--transport", "http", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test MCP protocol
        mcp_ok = await test_mcp_protocol()
        
        # Test Claude Desktop compatibility
        claude_ok = await test_claude_desktop_compatibility()
        
        print("\n" + "=" * 40)
        print("📊 Results:")
        print(f"   MCP Protocol: {'✅ PASS' if mcp_ok else '❌ FAIL'}")
        print(f"   Claude Desktop: {'✅ PASS' if claude_ok else '❌ FAIL'}")
        
        if mcp_ok and claude_ok:
            print("\n🎉 Your server should work with Claude!")
            print("📝 Next steps:")
            print("   1. Deploy to Railway with HTTPS")
            print("   2. Add server URL to Claude Desktop")
            print("   3. Complete OAuth flow")
        else:
            print("\n⚠️ Some issues found. Check the errors above.")
            
    finally:
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

if __name__ == "__main__":
    asyncio.run(main()) 