import urllib.request
import re

# Check LibriSpeech sizes
url = 'https://www.openslr.org/resources/12/'
try:
    resp = urllib.request.urlopen(url, timeout=10000)
    content = resp.read().decode('utf-8', errors='ignore')
    
    # Find dev-clean and test-clean
    for name in ['dev-clean', 'test-clean']:
        matches = re.findall(rf'href=["\']([^\"\']*{name}[^"\']*)["\'|>\s]*.*?(\d+\.?\d*\s*[MG])', content, re.IGNORECASE)
        for m in matches[:5]:
            print(f"{name}: {m[0]} -> {m[1]}")
except Exception as e:
    print(f"Error: {e}")
