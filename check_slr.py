import urllib.request
import re

urls = {
    'LibriSpeech': 'https://www.openslr.org/resources/12/',
    'TIMIT': 'https://www.openslr.org/resources/11/',
    'VoxForge': 'https://www.openslr.org/resources/17/',
    'LibriSpeech segments': 'https://www.openslr.org/resources/47/',
}

for name, url in urls.items():
    try:
        resp = urllib.request.urlopen(url, timeout=8000)
        content = resp.read().decode('utf-8', errors='ignore')
        title = re.search(r'<title>(.*?)</title>', content)
        title_str = title.group(1) if title else 'unknown'
        # Find size hints
        sizes = re.findall(r'>(\d+\.?\d*\s*[MG])</a>', content)
        print(f'{name}: {title_str}')
        if sizes:
            print(f'  Sizes: {sizes[:5]}')
    except Exception as e:
        print(f'FAIL {name}: {e}')
