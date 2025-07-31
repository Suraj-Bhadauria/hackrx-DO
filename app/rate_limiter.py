import time
import asyncio
from typing import List, Dict, Tuple

class MultiKeyRateLimiter:
    """Smart rate limiter for multiple Groq API keys with TRUE round-robin distribution"""
    
    def __init__(self, api_keys: List[str], max_requests_per_minute=25):
        self.api_keys = api_keys
        self.max_rpm = max_requests_per_minute
        self.key_count = len(api_keys)
        
        # Track requests per key
        self.key_request_times: Dict[int, List[float]] = {
            i: [] for i in range(self.key_count)
        }
        
        # Round-robin counter - THIS IS CRITICAL FOR DISTRIBUTION
        self.current_key_index = 0
        self.lock = asyncio.Lock()
        
        print(f"🔥 Multi-Key Rate Limiter initialized with {self.key_count} keys")
        print(f"⚡ Total capacity: {self.key_count * self.max_rpm} RPM")
        
    async def get_next_available_key(self) -> Tuple[str, int]:
        """Get next available API key using TRUE round-robin distribution"""
        async with self.lock:
            attempts = 0
            max_attempts = self.key_count * 2
            
            while attempts < max_attempts:
                # Get current key using round-robin
                key_index = self.current_key_index
                api_key = self.api_keys[key_index]
                
                # CRITICAL: Move to next key IMMEDIATELY for true distribution
                self.current_key_index = (self.current_key_index + 1) % self.key_count
                
                # Check if this key can handle another request
                now = time.time()
                cutoff_time = now - 60
                
                # Clean old requests for this key
                self.key_request_times[key_index] = [
                    t for t in self.key_request_times[key_index] 
                    if t > cutoff_time
                ]
                
                # Check if key is available
                if len(self.key_request_times[key_index]) < self.max_rpm:
                    # Record this request
                    self.key_request_times[key_index].append(now)
                    
                    key_requests = len(self.key_request_times[key_index])
                    print(f"🔑 Using API Key #{key_index + 1} (ends with: ...{api_key[-8:]})")
                    print(f"✅ Request {key_requests}/{self.max_rpm} this minute")
                    
                    return api_key, key_index
                
                attempts += 1
            
            # If all keys are rate limited, wait for the best one
            return await self._wait_for_best_key()
    
    async def _wait_for_best_key(self) -> Tuple[str, int]:
        """Wait for the key that will be available soonest"""
        now = time.time()
        best_key_index = 0
        best_wait_time = float('inf')
        
        for key_index in range(self.key_count):
            if self.key_request_times[key_index]:
                oldest_request = min(self.key_request_times[key_index])
                wait_time = 61 - (now - oldest_request)
                if wait_time < best_wait_time:
                    best_wait_time = wait_time
                    best_key_index = key_index
        
        if best_wait_time > 0:
            print(f"⏳ All keys rate limited - waiting {best_wait_time:.1f}s for Key #{best_key_index + 1}")
            await asyncio.sleep(best_wait_time)
        
        # Record request and return
        self.key_request_times[best_key_index].append(time.time())
        api_key = self.api_keys[best_key_index]
        return api_key, best_key_index

class SingleKeyRateLimiter:
    """Legacy single-key rate limiter with compatibility method"""
    
    def __init__(self, api_key: str, max_requests_per_minute=25):
        self.api_key = api_key
        self.max_rpm = max_requests_per_minute
        self.request_times: List[float] = []
        self.lock = asyncio.Lock()
        
    async def get_next_available_key(self) -> Tuple[str, int]:
        """Compatibility method for single key"""
        await self.acquire()
        return self.api_key, 0
        
    async def acquire(self):
        """Wait if needed to respect rate limits"""
        async with self.lock:
            now = time.time()
            cutoff_time = now - 60
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            
            if len(self.request_times) >= self.max_rpm:
                oldest_request = self.request_times[0]
                wait_time = 61 - (now - oldest_request)
                
                if wait_time > 0:
                    print(f"⏳ Rate limiting: waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    
                    now = time.time()
                    cutoff_time = now - 60
                    self.request_times = [t for t in self.request_times if t > cutoff_time]
            
            self.request_times.append(now)
            print(f"✅ Request {len(self.request_times)}/{self.max_rpm} this minute")

# FIXED: Proper initialization
def create_rate_limiter():
    """Factory function to create the appropriate rate limiter"""
    from . import config
    
    if hasattr(config, 'GROQ_API_KEYS') and len(config.GROQ_API_KEYS) > 1:
        print(f"🚀 Creating Multi-Key Rate Limiter with {len(config.GROQ_API_KEYS)} keys")
        return MultiKeyRateLimiter(config.GROQ_API_KEYS)
    elif hasattr(config, 'GROQ_API_KEYS') and len(config.GROQ_API_KEYS) == 1:
        print("🔄 Creating Single-Key Rate Limiter")
        return SingleKeyRateLimiter(config.GROQ_API_KEYS[0])
    elif hasattr(config, 'GROQ_API_KEY') and config.GROQ_API_KEY:
        print("🔄 Creating Single-Key Rate Limiter (legacy)")
        return SingleKeyRateLimiter(config.GROQ_API_KEY)
    else:
        raise ValueError("No API keys available!")

# Global instance
_global_rate_limiter = None

def get_rate_limiter():
    """Get or create the global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = create_rate_limiter()
    return _global_rate_limiter

# BACKWARD COMPATIBILITY - this is what your current code imports
rate_limiter = None  # Will be lazy-loaded when first used