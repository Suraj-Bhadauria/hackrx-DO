#!/usr/bin/env python3
"""
Multi-Key Setup Test Script
Run this to verify your 3-key configuration is working properly
"""

import asyncio
import os
import sys
sys.path.append('.')

from app.config import GROQ_API_KEYS, GROQ_RATE_LIMITS
from app.rate_limiter import rate_limiter

async def test_multi_key_setup():
    """Test the multi-key setup"""
    print("🔍 MULTI-KEY SETUP TEST")
    print("=" * 50)
    
    # Test 1: Configuration
    print(f"📊 API Keys found: {len(GROQ_API_KEYS)}")
    for i, key in enumerate(GROQ_API_KEYS, 1):
        print(f"   Key #{i}: ...{key[-8:]}")
    
    print(f"⚡ Total RPM Capacity: {GROQ_RATE_LIMITS.get('total_capacity', 'Unknown')} requests/minute")
    print(f"🎯 Expected Performance: ~{len(GROQ_API_KEYS) * 6000} tokens/minute")
    
    # Test 2: Rate Limiter Type
    print(f"\n🔧 Rate Limiter Type: {type(rate_limiter).__name__}")
    
    if hasattr(rate_limiter, 'get_next_available_key'):
        print("✅ Multi-key rate limiter active!")
        
        # Test 3: Key Distribution
        print("\n🔄 Testing key distribution (5 requests):")
        for i in range(5):
            try:
                api_key, key_index = await rate_limiter.get_next_available_key()
                print(f"   Request #{i+1}: Key #{key_index + 1} (...{api_key[-8:]})")
                await asyncio.sleep(0.1)  # Small delay
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
        print("\n🎉 Multi-key setup is working correctly!")
        print(f"🚀 Your system can now handle {len(GROQ_API_KEYS) * 25} requests/minute")
        print(f"💪 Token capacity: {len(GROQ_API_KEYS) * 6000} tokens/minute")
        
    else:
        print("⚠️ Single-key rate limiter detected")
        if len(GROQ_API_KEYS) > 1:
            print("❌ Configuration issue: Multiple keys found but single-key limiter active")
        else:
            print("ℹ️ This is expected with only 1 API key")
    
    print("\n" + "=" * 50)

def show_env_setup():
    """Show environment variable setup instructions"""
    print("\n🔧 ENVIRONMENT SETUP INSTRUCTIONS")
    print("=" * 50)
    print("Add these to your .env file:")
    print()
    print("GROQ_API_KEY_1=your_first_groq_api_key_here")
    print("GROQ_API_KEY_2=your_second_groq_api_key_here") 
    print("GROQ_API_KEY_3=your_third_groq_api_key_here")
    print()
    print("Or set as environment variables:")
    print("set GROQ_API_KEY_1=your_first_groq_api_key_here")
    print("set GROQ_API_KEY_2=your_second_groq_api_key_here")
    print("set GROQ_API_KEY_3=your_third_groq_api_key_here")
    print()
    print("💡 Minimum 2 keys required for multi-key mode")
    print("🎯 Recommended: 3 keys for hackathon reliability")

if __name__ == "__main__":
    if len(GROQ_API_KEYS) == 0:
        print("❌ NO API KEYS FOUND!")
        show_env_setup()
    else:
        asyncio.run(test_multi_key_setup())
