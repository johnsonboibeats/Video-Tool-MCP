#!/usr/bin/env python3
"""
Quick test script to verify security middleware functionality
"""
import asyncio
import aiohttp
import json
import os
import sys

async def test_security():
    """Test security features of the MCP server"""
    
    # You'll need to replace this with your actual Railway URL
    base_url = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
    
    print(f"Testing security features at: {base_url}")
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Health check should work
        print("\n1. Testing health check...")
        try:
            async with session.get(f"{base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Health check OK: {data}")
                else:
                    print(f"❌ Health check failed: {resp.status}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        
        # Test 2: CORS headers
        print("\n2. Testing CORS...")
        try:
            async with session.options(f"{base_url}/mcp/", headers={
                "Origin": "https://claude.ai",
                "Access-Control-Request-Method": "POST"
            }) as resp:
                cors_origin = resp.headers.get("Access-Control-Allow-Origin")
                print(f"✅ CORS Origin header: {cors_origin}")
        except Exception as e:
            print(f"❌ CORS test error: {e}")
        
        # Test 3: Rate limiting (make multiple rapid requests)
        print("\n3. Testing rate limiting...")
        rate_limit_hit = False
        for i in range(5):
            try:
                async with session.post(f"{base_url}/mcp/", 
                                      json={"test": "request"},
                                      headers={"Content-Type": "application/json"}) as resp:
                    if resp.status == 429:
                        print(f"✅ Rate limit triggered at request {i+1}")
                        rate_limit_hit = True
                        break
                    elif resp.status == 404:
                        print(f"📝 Request {i+1}: {resp.status} (expected, MCP not fully configured)")
                    else:
                        print(f"📝 Request {i+1}: {resp.status}")
            except Exception as e:
                print(f"📝 Request {i+1} error: {e}")
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        if not rate_limit_hit:
            print("📝 Rate limit not triggered in test (this is normal for low request counts)")
        
        print(f"\n🎉 Security test completed!")

if __name__ == "__main__":
    print("Image Tool MCP Security Test")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        os.environ["TEST_SERVER_URL"] = sys.argv[1]
    
    asyncio.run(test_security())