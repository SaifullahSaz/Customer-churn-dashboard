#!/usr/bin/env python3
"""scripts/test_supabase.py

Simple local checker for Supabase connectivity and table visibility.

Usage:
  python scripts/test_supabase.py            # reads .streamlit/secrets.toml
  python scripts/test_supabase.py --url URL --table TABLE --key KEY

This script is intended for local development only and will print clear
diagnostics for DNS resolution, HTTP reachability, and a small REST query
against the project's PostgREST endpoint (/rest/v1/<table>?limit=1).
"""
import sys
import os
import socket
import urllib.parse
import argparse

try:
    import tomllib
except Exception:  # Python <3.11 fallback (unlikely in this workspace)
    try:
        import tomli as tomllib
    except Exception:
        tomllib = None

try:
    import requests
except Exception:
    requests = None


def load_secrets(path=".streamlit/secrets.toml"):
    if not tomllib:
        print("tomllib/tomli is not available in this Python environment. Install tomli or use Python>=3.11.")
        return {}
    if not os.path.exists(path):
        print(f"Secrets file not found at {path}. Create it or pass --url/--key/--table via CLI.")
        return {}
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data


def dns_check(host):
    print(f"Resolving host: {host}")
    try:
        infos = socket.getaddrinfo(host, 443)
        addrs = sorted({info[4][0] for info in infos})
        print("Resolved addresses:")
        for a in addrs:
            print(" ", a)
        return True
    except Exception as e:
        print("DNS resolution failed:", e)
        return False


def http_check(url):
    print(f"HTTP HEAD {url}")
    if requests is None:
        print("requests library not installed. Install with: pip install requests")
        return False
    try:
        r = requests.head(url, timeout=8)
        print("Status:", r.status_code)
        return True
    except Exception as e:
        print("HTTP head request failed:", e)
        return False


def rest_check(url, table, key=None):
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}?limit=1"
    print(f"Testing REST endpoint: {endpoint}")
    if requests is None:
        print("requests library not installed; cannot run REST test.")
        return
    headers = {}
    if key:
        headers["apikey"] = key
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = requests.get(endpoint, headers=headers, timeout=10)
        print("Status code:", r.status_code)
        try:
            print("Response (truncated):", r.text[:1000])
        except Exception:
            pass
        if r.status_code >= 400:
            print("REST request returned an error. If the message contains PGRST205, the table name may be incorrect or not visible to this role.")
    except Exception as e:
        print("REST request failed:", e)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--secrets", default=".streamlit/secrets.toml", help="Path to secrets.toml")
    p.add_argument("--url", help="Supabase URL override (https://<project>.supabase.co)")
    p.add_argument("--table", help="Supabase table name override")
    p.add_argument("--key", help="Supabase API key override (optional, used for REST test)")
    args = p.parse_args(argv)

    secrets = load_secrets(args.secrets) or {}

    sup = secrets.get("supabase", {})
    url = args.url or sup.get("url")
    table = args.table or sup.get("table")
    key = args.key or sup.get("key")

    if not url:
        print("No Supabase URL provided. Set it in .streamlit/secrets.toml or pass --url.")
        sys.exit(2)
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc or parsed.path

    print("--- Supabase quick test ---")
    print("URL:", url)
    print("Table:", table)
    print("Key present:", bool(key))

    ok = dns_check(host)
    if not ok:
        print("DNS failed — check your network, VPN, proxy, or the URL value in secrets.")

    http_check(url)

    if table:
        rest_check(url, table, key=key)
    else:
        print("No table provided; skipping REST endpoint test.")


if __name__ == "__main__":
    main()
