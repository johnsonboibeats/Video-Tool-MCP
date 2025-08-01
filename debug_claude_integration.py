#!/usr/bin/env python3
"""
Debug script to check why tools aren't showing in Claude
"""

import asyncio
import httpx
import json
import subprocess
import time
import sys
from pathlib import Path

# Test configuration
BASE_URL = "http://localhost:8000"
SERVER_URL = "http://localhost:8000/mcp/"

async def test_server_health():
    """Test basic server health"""
    print("🏥 Testing server health...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health endpoint working")
                print(f"   Status: {health_data.get('status')}")
                print(f"   OAuth enabled: {health_data.get('oauth_enabled')}")
                return True
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

async def test_mcp_endpoint():
    """Test MCP endpoint directly"""
    print("\n🔧 Testing MCP endpoint...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test MCP endpoint with proper headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            # Test initialization
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
            
            response = await client.post(f"{SERVER_URL}", json=init_request, headers=headers)
            print(f"   Init response status: {response.status_code}")
            
            if response.status_code == 200:
                init_data = response.json()
                print("✅ MCP initialization successful")
                print(f"   Server name: {init_data.get('result', {}).get('serverInfo', {}).get('name')}")
                return True
            else:
                print(f"❌ MCP initialization failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ MCP endpoint error: {e}")
            return False

async def test_tools_listing():
    """Test tools listing"""
    print("\n🛠️ Testing tools listing...")
    
    async with httpx.AsyncClient() as client:
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            # List tools request
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            response = await client.post(f"{SERVER_URL}", json=tools_request, headers=headers)
            print(f"   Tools list response status: {response.status_code}")
            
            if response.status_code == 200:
                tools_data = response.json()
                tools = tools_data.get('result', {}).get('tools', [])
                print(f"✅ Tools listing successful - Found {len(tools)} tools")
                
                for tool in tools:
                    print(f"   - {tool.get('name')}: {tool.get('description', 'No description')}")
                
                return len(tools) > 0
            else:
                print(f"❌ Tools listing failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Tools listing error: {e}")
            return False

async def test_oauth_discovery():
    """Test OAuth discovery"""
    print("\n🔐 Testing OAuth discovery...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Test OAuth discovery endpoint
            response = await client.get(f"{BASE_URL}/.well-known/oauth-authorization-server")
            if response.status_code == 200:
                oauth_data = response.json()
                print("✅ OAuth discovery working")
                print(f"   Issuer: {oauth_data.get('issuer')}")
                print(f"   Supported scopes: {oauth_data.get('scopes_supported')}")
                return True
            else:
                print(f"❌ OAuth discovery failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ OAuth discovery error: {e}")
            return False

def check_server_configuration():
    """Check server configuration"""
    print("\n⚙️ Checking server configuration...")
    
    # Check if server.py exists
    if not Path("server.py").exists():
        print("❌ server.py not found")
        return False
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Check server.py for tool definitions
    with open("server.py", "r") as f:
        content = f.read()
        
    tool_count = content.count("@mcp.tool()")
    print(f"✅ Found {tool_count} tool definitions in server.py")
    
    # Check for common issues
    issues = []
    
    if "@mcp.tool()" not in content:
        issues.append("No tool decorators found")
    
    if "FastMCP" not in content:
        issues.append("FastMCP not imported")
    
    if "mcp.run" not in content:
        issues.append("Server run method not found")
    
    if issues:
        print("⚠️ Potential issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Server configuration looks good")
    return True

def check_claude_integration():
    """Check Claude integration requirements"""
    print("\n🤖 Checking Claude integration requirements...")
    
    requirements = [
        "HTTPS endpoint (for production)",
        "OAuth discovery endpoints",
        "Proper CORS configuration",
        "Tool definitions with proper schemas",
        "Health check endpoint"
    ]
    
    print("Claude integration requires:")
    for req in requirements:
        print(f"   ✅ {req}")
    
    print("\nCommon issues with Claude integration:")
    print("   1. Server not accessible via HTTPS")
    print("   2. OAuth flow not completed")
    print("   3. Tools not properly defined")
    print("   4. CORS issues")
    print("   5. Network connectivity problems")

async def test_full_integration():
    """Test full integration flow"""
    print("\n🔄 Testing full integration flow...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Step 1: Health check
            health_response = await client.get(f"{BASE_URL}/health")
            if health_response.status_code != 200:
                print("❌ Health check failed")
                return False
            
            # Step 2: OAuth discovery
            oauth_response = await client.get(f"{BASE_URL}/.well-known/oauth-authorization-server")
            if oauth_response.status_code != 200:
                print("❌ OAuth discovery failed")
                return False
            
            # Step 3: MCP initialization
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "Claude", "version": "1.0"}
                }
            }
            
            mcp_response = await client.post(f"{SERVER_URL}", json=init_request, headers=headers)
            if mcp_response.status_code != 200:
                print("❌ MCP initialization failed")
                return False
            
            # Step 4: Tools listing
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            tools_response = await client.post(f"{SERVER_URL}", json=tools_request, headers=headers)
            if tools_response.status_code != 200:
                print("❌ Tools listing failed")
                return False
            
            tools_data = tools_response.json()
            tools = tools_data.get('result', {}).get('tools', [])
            
            if len(tools) == 0:
                print("❌ No tools found")
                return False
            
            print(f"✅ Full integration successful - {len(tools)} tools available")
            return True
            
        except Exception as e:
            print(f"❌ Integration test error: {e}")
            return False

async def main():
    """Run all debugging tests"""
    print("🔍 Claude Integration Debugging")
    print("=" * 50)
    
    # Check server configuration
    config_ok = check_server_configuration()
    
    # Start server for testing
    print("\n🚀 Starting server for testing...")
    server_process = subprocess.Popen(
        ["python3", "server.py", "--transport", "http", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Run all tests
        tests = [
            ("Server Health", test_server_health),
            ("OAuth Discovery", test_oauth_discovery),
            ("MCP Endpoint", test_mcp_endpoint),
            ("Tools Listing", test_tools_listing),
            ("Full Integration", test_full_integration)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} test failed: {e}")
                results.append((test_name, False))
        
        # Check Claude integration requirements
        check_claude_integration()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        
        all_passed = True
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
            if not result:
                all_passed = False
        
        print("\n🎯 Recommendations:")
        
        if all_passed:
            print("   ✅ All tests passed! Your server should work with Claude.")
            print("   📝 Make sure to:")
            print("      - Deploy to HTTPS endpoint")
            print("      - Complete OAuth flow in Claude")
            print("      - Add server URL in Claude settings")
        else:
            print("   ⚠️ Some tests failed. Check the issues above.")
            print("   🔧 Common fixes:")
            print("      - Ensure server is running on port 8000")
            print("      - Check tool definitions in server.py")
            print("      - Verify OAuth endpoints are working")
            print("      - Test with Claude Desktop first")
        
    finally:
        # Clean up
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

if __name__ == "__main__":
    asyncio.run(main()) 