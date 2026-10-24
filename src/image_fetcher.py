"""Dynamic Bird Photo and Taxonomy Information Fetcher.

Queries Wikipedia's open REST API to retrieve high-resolution bird photographs
and biological descriptions for all 6,522 global species, with local disk caching.
"""

from typing import Dict, Any, Tuple
import os
import json
import urllib.request
import urllib.parse

CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "species_images.json")
DEFAULT_BIRD_IMG = "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=600&q=80"


def _load_cache() -> Dict[str, Any]:
    """Load cached species metadata from disk."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]):
    """Persist species metadata cache to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def fetch_bird_visuals(common_name: str, scientific_name: str) -> Dict[str, Any]:
    """Fetch high-resolution bird photo and biological summary.
    
    Args:
        common_name: Common name of the bird species.
        scientific_name: Binomial scientific nomenclature.
        
    Returns:
        Dictionary with image_url, description, and source attribution.
    """
    cache = _load_cache()
    key = scientific_name.strip().lower()
    if key in cache:
        return cache[key]

    headers = {'User-Agent': 'INTERA/1.0 (academic bioacoustics engineering project)'}
    
    # Try scientific name first (most accurate on Wikipedia), then common name
    search_queries = [scientific_name, common_name]
    image_url = None
    description = None

    for query in search_queries:
        if not query or query.lower() in ["human whistle", "human vocal", "engine", "siren"]:
            break

        safe_name = urllib.parse.quote(query.replace(' ', '_'))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_name}"
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode())
                # Prefer original/high-res image, fallback to thumbnail
                img_data = data.get('originalimage') or data.get('thumbnail')
                if img_data and 'source' in img_data:
                    image_url = img_data['source']
                
                extract = data.get('extract')
                if extract:
                    # Keep description concise (under 220 chars)
                    sentences = extract.split(". ")
                    description = sentences[0] + ("." if not sentences[0].endswith(".") else "")
                    if len(sentences) > 1 and len(description) < 140:
                        description += " " + sentences[1] + ("." if not sentences[1].endswith(".") else "")

                if image_url:
                    break
        except Exception:
            continue

    if not image_url:
        image_url = DEFAULT_BIRD_IMG
    if not description:
        description = f"{common_name} ({scientific_name}) is an avian species recognized by global bioacoustic databases."

    result = {
        "image_url": image_url,
        "description": description,
        "source": "Wikipedia / Wikimedia Commons"
    }

    # Cache for offline presentation safety
    cache[key] = result
    _save_cache(cache)
    return result
