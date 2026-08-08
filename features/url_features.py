"""
URL feature extraction module.
Extracts 18 lexical features from a URL string for machine learning classification.
"""

import ipaddress
import re
from urllib.parse import urlparse

import pandas as pd
import tldextract


# Feature column names expected by the trained ML model.
# Note: Feature names match the exact dataset schema stored in model artifacts.
FEATURE_COLS = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]

# Recognized trusted TLD suffixes (UK academic, gov, European, and international)
TRUSTED_TLDS = {
    "ac.uk", "co.uk", "gov.uk", "nhs.uk", "org.uk", "police.uk", "sch.uk", "me.uk", "net.uk", "uk",
    "edu", "gov", "mil", "int", "org",
    "de", "fr", "nl", "se", "no", "dk", "fi", "at", "ch", "be", "pt", "ie", "es", "it", "eu",
    "edu.au", "ac.nz", "ac.jp", "edu.sg", "edu.cn",
}


def add_scheme(url):
    """Ensure URL has a scheme (http://) so urlparse functions properly."""
    url = str(url).strip()
    if not url:
        raise ValueError("URL cannot be empty.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def is_ip(host):
    """Check if the domain hostname is a raw IP address."""
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0


def count_percent_encoded(url):
    """Count percent-encoded sequences (e.g. %20, %3A) in the URL."""
    return len(re.findall(r"%[0-9a-fA-F]{2}", url))


def is_known_tld(tld):
    """Check if the TLD suffix is in the trusted domain list."""
    return tld in TRUSTED_TLDS


def is_known_url_tld(url):
    """Check if a URL string belongs to a trusted TLD domain family."""
    try:
        host = (urlparse(add_scheme(str(url).strip())).hostname or "").lower()
        tld = tldextract.extract(host).suffix
        return is_known_tld(tld)
    except Exception:
        return False


def get_url_features(url):
    """
    Extract 18 lexical features from a URL string and return a 1-row DataFrame.
    """
    raw_url = str(url).strip()
    parsed = urlparse(add_scheme(raw_url))

    host = (parsed.hostname or "").lower()
    ext = tldextract.extract(host)
    tld = ext.suffix or ""

    total_len = len(raw_url)
    safe_len = max(total_len, 1)

    subdomains = [s for s in ext.subdomain.split(".") if s]
    num_encoded = count_percent_encoded(raw_url)
    num_letters = sum(c.isalpha() for c in raw_url)
    num_digits = sum(c.isdigit() for c in raw_url)

    # Count unusual and special characters
    normal_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_/:%?=&")
    num_unusual = sum(c not in normal_chars for c in raw_url)
    num_special = sum(not c.isalnum() for c in raw_url)

    features = {
        "URLLength": total_len,
        "DomainLength": len(host),
        "IsDomainIP": is_ip(host),
        "TLDLength": len(tld),
        "NoOfSubDomain": len(subdomains),
        "HasObfuscation": int(num_encoded > 0),
        "NoOfObfuscatedChar": num_encoded,
        "ObfuscationRatio": num_encoded / safe_len,
        "NoOfLettersInURL": num_letters,
        "LetterRatioInURL": num_letters / safe_len,
        "NoOfDegitsInURL": num_digits,
        "DegitRatioInURL": num_digits / safe_len,
        "NoOfEqualsInURL": raw_url.count("="),
        "NoOfQMarkInURL": raw_url.count("?"),
        "NoOfAmpersandInURL": raw_url.count("&"),
        "NoOfOtherSpecialCharsInURL": num_unusual,
        "SpacialCharRatioInURL": num_special / safe_len,
        "IsHTTPS": int(parsed.scheme.lower() == "https"),
    }

    return pd.DataFrame([features], columns=FEATURE_COLS)
