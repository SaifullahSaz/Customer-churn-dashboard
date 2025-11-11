#!/usr/bin/env python3
"""scripts/upsert_test_supabase.py

Insert/upsert a small test row into the Supabase `predictions` table using the
credentials in `.streamlit/secrets.toml`. This is intended for local debugging
and will create a unique `customer_id` to avoid accidental collisions.
"""
import os
import time
import json
import datetime
import urllib.parse
import sys

try:
    import tomllib
except Exception:
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
        print("tomllib/tomli not available. Install tomli or use Python>=3.11.")
        return {}
    if not os.path.exists(path):
        print(f"Secrets file not found at {path}")
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    if requests is None:
        print("Please install requests: pip install requests")
        sys.exit(2)

    secrets = load_secrets()
    sup = secrets.get("supabase", {})
    url = sup.get("url")
    key = sup.get("key")
    table = sup.get("table") or "predictions"

    if not url or not key:
        print("Missing supabase.url or supabase.key in .streamlit/secrets.toml")
        sys.exit(2)

    now = datetime.datetime.utcnow()
    unique_cust = f"test-{int(time.time())}-{os.getpid()}"
    record = {
        "customer_id": unique_cust,
        "predicted_churn": 0.12345,
        "monthly_charges": 42.0,
        "tenure": 1,
        "upload_time": now.isoformat(),
    }

    endpoint = f"{url.rstrip('/')}/rest/v1/{table}?on_conflict=customer_id"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation,resolution=merge-duplicates",
    }

    print("Upserting test record to:", endpoint)
    print("Record:", record)

    try:
        r = requests.post(endpoint, headers=headers, data=json.dumps([record]), timeout=15)
        print("Status code:", r.status_code)
        print("Response:", r.text[:2000])
        if r.status_code in (200, 201):
            print("Upsert successful.")
        else:
            # If ON CONFLICT fails because there is no unique constraint, try a plain insert
            text = r.text or ""
            if "ON CONFLICT" in text or "there is no unique or exclusion constraint" in text or r.status_code == 400:
                print("ON CONFLICT upsert failed; attempting a plain INSERT without on_conflict...")
                try:
                    fallback_endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
                    r2 = requests.post(fallback_endpoint, headers={k: v for k, v in headers.items() if k != "Prefer"}, data=json.dumps([record]), timeout=15)
                    print("Fallback status:", r2.status_code)
                    print("Fallback response:", r2.text[:2000])
                    if r2.status_code in (200, 201):
                        print("Insert successful (fallback).")
                    else:
                        print("Fallback insert also failed. Check table schema, NOT NULL constraints, and permissions.")
                except Exception as e2:
                    print("Fallback request failed:", e2)
            else:
                print("Upsert may have failed. Check the response above for PGRST errors or permission issues.")
    except Exception as e:
        print("Request failed:", e)


if __name__ == "__main__":
    main()
