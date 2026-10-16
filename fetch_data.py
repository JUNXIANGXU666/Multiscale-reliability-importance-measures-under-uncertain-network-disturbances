"""Download and verify the three public Sioux Falls benchmark inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/sioux_falls'


def sources():
    return json.loads((ROOT/'data/sources.json').read_text(encoding='utf-8'))['files']


def verify_data():
    for item in sources():
        path = DATA/item['name']
        if not path.is_file():
            raise FileNotFoundError('Run python fetch_data.py first. Missing '+item['name'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Unexpected data version: '+item['name']+
                             '. Consult data/sources.json and the upstream repository.')


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    for item in sources():
        path = DATA/item['name']
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Existing file differs from the reference input: '+str(path))
            print('Verified '+item['name'])
            continue
        request = Request(item['url'], headers={'User-Agent': 'research-data-download'})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != item['sha256']:
            raise ValueError('Upstream data changed: '+item['name']+'. No file was saved.')
        path.write_bytes(content)
        print('Downloaded and verified '+item['name'])
    verify_data()


if __name__ == '__main__':
    main()
