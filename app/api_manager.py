# app/api_manager.py

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from groq import AsyncGroq, GroqError
from . import config

class APIKeyManager:
    """
    Advanced API Key management with health checking and automatic failover
    """
    
    def __init__(self):
        self.api_keys = config.GROQ_API_KEYS.copy()
        self.key_status = {}  # Track health of each key
        self.key_usage_count = {}  # Track usage per key
        self.blocked_keys = set()  # Track blocked keys
        self.last_health_check = None
        
        # Initialize tracking
        for i, key in enumerate(self.api_keys):
            self.key_status[i] = {
                'healthy': True,
                'last_error': None,
                'error_count': 0,
                'last_success': datetime.now()
            }
            self.key_usage_count[i] = 0
    
    def get_healthy_keys(self) -> List[tuple]:
        """Get list of healthy API keys with their indices"""
        healthy = []
        for i, key in enumerate(self.api_keys):
            if i not in self.blocked_keys and self.key_status[i]['healthy']:
                healthy.append((i, key))
        return healthy
    
    def mark_key_error(self, key_index: int, error: str):
        """Mark a key as having an error"""
        if key_index in self.key_status:
            self.key_status[key_index]['error_count'] += 1
            self.key_status[key_index]['last_error'] = error
            
            # Mark as blocked if organization restricted
            if 'organization_restricted' in error.lower() or 'restricted' in error.lower():
                self.blocked_keys.add(key_index)
                print(f"🚫 API Key #{key_index + 1} marked as BLOCKED: {error}")
            
            # Mark as unhealthy if too many errors
            if self.key_status[key_index]['error_count'] >= 3:
                self.key_status[key_index]['healthy'] = False
                print(f"⚠️ API Key #{key_index + 1} marked as UNHEALTHY after {self.key_status[key_index]['error_count']} errors")
    
    def mark_key_success(self, key_index: int):
        """Mark a key as working successfully"""
        if key_index in self.key_status:
            self.key_status[key_index]['healthy'] = True
            self.key_status[key_index]['last_success'] = datetime.now()
            self.key_usage_count[key_index] += 1
    
    def get_best_key(self) -> Optional[tuple]:
        """Get the best available API key (least used, healthy)"""
        healthy_keys = self.get_healthy_keys()
        
        if not healthy_keys:
            print("❌ No healthy API keys available!")
            return None
        
        # Sort by usage count (least used first)
        healthy_keys.sort(key=lambda x: self.key_usage_count[x[0]])
        return healthy_keys[0]
    
    async def test_key_health(self, key_index: int, api_key: str) -> bool:
        """Test if an API key is working"""
        try:
            client = AsyncGroq(api_key=api_key)
            response = await client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                timeout=10
            )
            
            if response and response.choices:
                self.mark_key_success(key_index)
                return True
                
        except Exception as e:
            self.mark_key_error(key_index, str(e))
            return False
        
        return False
    
    async def health_check_all_keys(self):
        """Check health of all API keys"""
        print("🔍 Running API key health check...")
        
        tasks = []
        for i, key in enumerate(self.api_keys):
            if i not in self.blocked_keys:  # Don't test blocked keys
                task = self.test_key_health(i, key)
                tasks.append((i, task))
        
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (key_index, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    self.mark_key_error(key_index, str(result))
                elif result:
                    print(f"✅ API Key #{key_index + 1} is healthy")
                else:
                    print(f"❌ API Key #{key_index + 1} failed health check")
        
        self.last_health_check = datetime.now()
    
    def print_status_report(self):
        """Print comprehensive status report"""
        print(f"\n{'='*60}")
        print(f"🔑 API KEY STATUS REPORT - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        healthy_count = len(self.get_healthy_keys())
        blocked_count = len(self.blocked_keys)
        total_keys = len(self.api_keys)
        
        print(f"📊 SUMMARY: {healthy_count}/{total_keys} keys healthy, {blocked_count} blocked")
        print(f"")
        
        for i, key in enumerate(self.api_keys):
            status = self.key_status[i]
            usage = self.key_usage_count[i]
            
            health_icon = "✅" if status['healthy'] else "❌"
            blocked_icon = "🚫" if i in self.blocked_keys else ""
            
            print(f"{health_icon} Key #{i+1} (...{key[-8:]}): {usage} uses {blocked_icon}")
            
            if status['last_error']:
                print(f"   Last Error: {status['last_error'][:80]}...")
            
            if status['error_count'] > 0:
                print(f"   Error Count: {status['error_count']}")
        
        print(f"{'='*60}\n")

# Global instance
api_manager = APIKeyManager()
