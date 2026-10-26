"""Dynamic Bird Photo and Taxonomy Information Fetcher.

Retrieves verified bird photographs and biological descriptions for global species,
with persistent disk caching, SSRF protection, strict URL validation, and zero-lag local asset fallback.
"""

import json
import logging
import os
import re
import urllib.parse
import urllib.request
from typing import Any
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "species_images.json")
PHOTOS_DIR = "data/species_photos"
DEFAULT_BIRD_IMG = "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=600&q=80"
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB download cap

# SSRF Mitigation: Allowed domain patterns
ALLOWED_DOMAINS = (
    "en.wikipedia.org",
    "upload.wikimedia.org",
    "thumb.wikimedia.org",
    "commons.wikimedia.org",
    "images.unsplash.com"
)


def _is_safe_url(url: str) -> bool:
    """Validate that a URL uses HTTPS and points strictly to trusted domains."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() != "https":
            return False
        hostname = (parsed.hostname or "").lower()
        return any(hostname == d or hostname.endswith("." + d) for d in ALLOWED_DOMAINS)
    except (ValueError, TypeError, AttributeError):
        return False


def _safe_slug(name: str) -> str:
    """Sanitize input string to prevent directory traversal attacks."""
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', name.strip().lower())
    return cleaned[:64] or "unknown_species"


def _load_cache() -> dict[str, Any]:
    """Load cached species metadata from disk safely."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cache read failed (%s); starting with empty cache.", e)
            return {}
    return {}


def _save_cache(cache: dict[str, Any]):
    """Persist species metadata cache to disk safely."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning("Cache write failed: %s", e)


def fetch_bird_visuals(common_name: str, scientific_name: str) -> dict[str, Any]:
    """Fetch high-resolution bird photo and biological summary safely.
    
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
        if os.path.exists(res.get("image_url", "")) or res.get("image_url", "").startswith("https://"):
            return res
    if key_com in cache:
        res = cache[key_com]
        if os.path.exists(res.get("image_url", "")) or res.get("image_url", "").startswith("https://"):
            return res

    # Check local photos directory with sanitized filename
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    for slug in [_safe_slug(key_sci), _safe_slug(key_com)]:
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

        safe_name = urllib.parse.quote(_safe_slug(query).replace('_', ' '))
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_name}"

        if not _is_safe_url(api_url):
            continue

        req = urllib.request.Request(api_url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:  # nosec B310 - URL validated by _is_safe_url
                data = json.loads(resp.read().decode())
                img_data = data.get('thumbnail') or data.get('originalimage')
                if img_data and 'source' in img_data:
                    raw_img_url = img_data['source']

                    if _is_safe_url(raw_img_url):
                        slug = _safe_slug(key_sci)
                        target_local = os.path.join(PHOTOS_DIR, f"{slug}.jpg")
                        try:
                            dl_req = urllib.request.Request(raw_img_url, headers=headers)
                            with urllib.request.urlopen(dl_req, timeout=4.0) as dl_resp:  # nosec B310 - URL validated by _is_safe_url
                                content = dl_resp.read(MAX_IMAGE_BYTES)
                                with open(target_local, "wb") as f_out:
                                    f_out.write(content)
                                image_url = target_local
                        except (URLError, HTTPError, OSError):
                            image_url = raw_img_url
                
                extract = data.get('extract')
                if extract:
                    sentences = extract.split(". ")
                    description = sentences[0] + ("." if not sentences[0].endswith(".") else "")
                    if len(sentences) > 1 and len(description) < 140:
                        description += " " + sentences[1] + ("." if not sentences[1].endswith(".") else "")

                if image_url:
                    break
        except (URLError, HTTPError, json.JSONDecodeError, OSError):
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
