#!/usr/bin/env python3
"""
Verify deployed server works correctly
"""

import asyncio
import httpx
import sys

async def verify_deployment(server_url: str):
    """Verify deployed server works correctly"""
    print(f"🔍 Verifying deployment: {server_url}")
    print("=" * 50)
    
    async with httpx.AsyncClient() as client:
        try:
            # Test health endpoint
            print("🏥 Testing health endpoint...")
            response = await client.get(f"{server_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health endpoint working")
                print(f"   Status: {health_data.get('status')}")
                print(f"   OAuth enabled: {health_data.get('oauth_enabled')}")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                return False
            
            # Test OAuth discovery
            print("\n🔐 Testing OAuth discovery...")
            response = await client.get(f"{server_url}/.well-known/oauth-authorization-server")
            if response.status_code == 200:
                oauth_data = response.json()
                print("✅ OAuth discovery working")
                print(f"   Issuer: {oauth_data.get('issuer')}")
                print(f"   Supported scopes: {oauth_data.get('scopes_supported')}")
            else:
                print(f"❌ OAuth discovery failed: {response.status_code}")
                return False
            
            # Test root endpoint
            print("\n🏠 Testing root endpoint...")
            response = await client.get(f"{server_url}/")
            if response.status_code == 200:
                print("✅ Root endpoint working")
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
                return False
            
            print("\n" + "=" * 50)
            print("🎉 Server verification successful!")
            print("\n📝 Next steps for Claude integration:")
            print("   1. Add this URL to Claude Desktop: Settings > Connectors")
            print("   2. Complete the OAuth authorization flow")
            print("   3. Your 12 image processing tools will be available")
            
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False

async def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 verify_deployment.py <server_url>")
        print("Example: python3 verify_deployment.py https://your-server.railway.app")
        return
    
    server_url = sys.argv[1]
    success = await verify_deployment(server_url)
    
    if not success:
        print("\n⚠️ Server verification failed. Check the issues above.")

if __name__ == "__main__":
    asyncio.run(main()) 