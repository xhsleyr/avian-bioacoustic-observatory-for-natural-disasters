"""Dynamic Bird Photo and Taxonomy Information Fetcher.

Retrieves verified bird photographs and biological descriptions for global species,
with persistent disk caching and zero-lag local asset fallback.
"""

from typing import Dict, Any, Tuple
import os
import json
import urllib.request
import urllib.parse

CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "species_images.json")
PHOTOS_DIR = "data/species_photos"
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
    key_sci = scientific_name.strip().lower()
    key_com = common_name.strip().lower()

    # Instant return from cache
    if key_sci in cache:
        res = cache[key_sci]
        if os.path.exists(res.get("image_url", "")) or res.get("image_url", "").startswith("http"):
            return res
    if key_com in cache:
        res = cache[key_com]
        if os.path.exists(res.get("image_url", "")) or res.get("image_url", "").startswith("http"):
            return res

    # Check local photos directory
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    for slug in [key_sci.replace(" ", "_"), key_com.replace(" ", "_")]:
        local_path = os.path.join(PHOTOS_DIR, f"{slug}.jpg")
        if os.path.exists(local_path):
            result = {
                "image_url": local_path,
                "description": f"{common_name} ({scientific_name}) is an avian species recognized by global bioacoustic databases.",
                "source": "Local Bioacoustic Image Archive"
            }
            cache[key_sci] = result
            cache[key_com] = result
            _save_cache(cache)
            return result

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)'
    }
    
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
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode())
                img_data = data.get('thumbnail') or data.get('originalimage')
                if img_data and 'source' in img_data:
                    raw_img_url = img_data['source']
                    # Attempt local download for 0ms offline presentation
                    slug = key_sci.replace(" ", "_")
                    target_local = os.path.join(PHOTOS_DIR, f"{slug}.jpg")
                    try:
                        dl_req = urllib.request.Request(raw_img_url, headers=headers)
                        with urllib.request.urlopen(dl_req, timeout=4.0) as dl_resp:
                            with open(target_local, "wb") as f_out:
                                f_out.write(dl_resp.read())
                            image_url = target_local
                    except Exception:
                        image_url = raw_img_url
                
                extract = data.get('extract')
                if extract:
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
        "source": "Wikipedia / Wikimedia Commons" if image_url != DEFAULT_BIRD_IMG else "Bioacoustic Reference Visual"
    }

    cache[key_sci] = result
    cache[key_com] = result
    _save_cache(cache)
    return result
