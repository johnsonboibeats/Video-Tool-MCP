#!/bin/bash

# Railway Deployment Script
echo "🚀 Railway Deployment Script"
echo "============================"

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "   npm install -g @railway/cli"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "railway.json" ]; then
    echo "❌ railway.json not found. Please run this script from the project root."
    exit 1
fi

# Check if we're logged in to Railway
if ! railway whoami &> /dev/null; then
    echo "❌ Not logged in to Railway. Please run:"
    echo "   railway login"
    exit 1
fi

echo "✅ Railway CLI is ready"

# Check current project status
echo ""
echo "📊 Current Project Status:"
railway status

# Check environment variables
echo ""
echo "🔧 Environment Variables:"
railway variables

# Deploy the application
echo ""
echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Set your OpenAI API key: railway variables set OPENAI_API_KEY=your_key"
echo "2. Check the deployment logs in Railway dashboard"
echo "3. Test the health endpoint: https://your-app.railway.app/health"
echo "4. Verify the MCP endpoint is accessible" 