"""Interactive Pinterest OAuth helper.

One-time setup that exchanges a Pinterest authorization code for an
access + refresh token pair and writes them to disk where
:class:`adapters.trends.pinterest_adapter.PinterestAdapter` will pick
them up automatically.

Quick start
-----------

1.  Register an app at https://developers.pinterest.com → My apps →
    Connect app. Set a redirect URI you control; ``http://localhost:8085/callback``
    works for local dev. Note the **App ID** and **App secret**.

2.  Put them in your environment::

        export PINTEREST_APP_ID=...
        export PINTEREST_APP_SECRET=...
        export PINTEREST_REDIRECT_URI="http://localhost:8085/callback"

3.  Run the *authorize* step::

        python -m scripts.pinterest_oauth authorize

    The script prints a URL. Open it in a browser, complete consent,
    and copy the ``code=`` query parameter from the redirect URL back
    into the script's prompt.

4.  The script writes the token to ``~/.margo/pinterest_token.json``
    (override with ``PINTEREST_TOKEN_FILE``). The adapter will use it
    on the next snapshot run automatically.

5.  Tokens expire after 60 days. Refresh in-place with::

        python -m scripts.pinterest_oauth refresh
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import requests


_AUTHORIZE_URL = "https://www.pinterest.com/oauth/"
_TOKEN_URL = "https://api.pinterest.com/v5/oauth/token"
_DEFAULT_SCOPE = "user_accounts:read"
_DEFAULT_TOKEN_FILE = "~/.margo/pinterest_token.json"


def _token_file_path() -> Path:
    return Path(os.path.expanduser(
        os.getenv("PINTEREST_TOKEN_FILE", _DEFAULT_TOKEN_FILE)
    ))


def _basic_auth_header(app_id: str, app_secret: str) -> str:
    encoded = base64.b64encode(f"{app_id}:{app_secret}".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


def _write_token(payload: dict, *, refresh_token: str | None = None) -> Path:
    """Persist the token response with a computed expiry timestamp."""
    expires_in = int(payload.get("expires_in", 0))
    expires_at = datetime.utcnow() + timedelta(seconds=expires_in) if expires_in else None
    data = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token", refresh_token),
        "expires_in": expires_in,
        "expires_at_iso": expires_at.isoformat() if expires_at else None,
        "scope": payload.get("scope"),
        "token_type": payload.get("token_type", "bearer"),
        "obtained_at_iso": datetime.utcnow().isoformat(),
    }
    path = _token_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def _require_app_credentials() -> tuple[str, str, str]:
    app_id = os.getenv("PINTEREST_APP_ID")
    app_secret = os.getenv("PINTEREST_APP_SECRET")
    redirect_uri = os.getenv("PINTEREST_REDIRECT_URI", "http://localhost:8085/callback")
    if not app_id or not app_secret:
        sys.exit(
            "PINTEREST_APP_ID / PINTEREST_APP_SECRET must be set. "
            "See https://developers.pinterest.com → My apps."
        )
    return app_id, app_secret, redirect_uri


def cmd_authorize(args: argparse.Namespace) -> None:
    app_id, app_secret, redirect_uri = _require_app_credentials()
    scope = args.scope or _DEFAULT_SCOPE

    params = {
        "client_id": app_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "state": "margo",
    }
    auth_url = f"{_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    print("\n1) Open this URL in a browser and complete consent:\n")
    print(f"   {auth_url}\n")
    print("2) After consent Pinterest will redirect to your redirect_uri")
    print(f"   ({redirect_uri}) with a ?code=... query param.")
    print("   Copy that code value and paste it below.\n")

    code = input("Paste the authorization code: ").strip()
    if not code:
        sys.exit("No code provided — aborting.")

    resp = requests.post(
        _TOKEN_URL,
        headers={
            "Authorization": _basic_auth_header(app_id, app_secret),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if resp.status_code >= 400:
        sys.exit(f"Token exchange failed (HTTP {resp.status_code}): {resp.text}")

    payload = resp.json()
    path = _write_token(payload)
    print(f"\n✓ token saved → {path}")
    print(f"  scope        = {payload.get('scope')}")
    print(f"  expires_in   = {payload.get('expires_in')}s")


def cmd_refresh(args: argparse.Namespace) -> None:
    app_id, app_secret, _ = _require_app_credentials()
    path = _token_file_path()
    if not path.exists():
        sys.exit(f"No existing token at {path}; run `authorize` first.")
    existing = json.loads(path.read_text(encoding="utf-8"))
    refresh_token = existing.get("refresh_token")
    if not refresh_token:
        sys.exit("Existing token file has no refresh_token; re-authorize from scratch.")

    resp = requests.post(
        _TOKEN_URL,
        headers={
            "Authorization": _basic_auth_header(app_id, app_secret),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    if resp.status_code >= 400:
        sys.exit(f"Refresh failed (HTTP {resp.status_code}): {resp.text}")

    payload = resp.json()
    # Pinterest returns a fresh access_token but may keep the same refresh_token.
    written = _write_token(payload, refresh_token=refresh_token)
    print(f"✓ token refreshed → {written}")


def cmd_show(args: argparse.Namespace) -> None:
    path = _token_file_path()
    if not path.exists():
        sys.exit(f"No token at {path}.")
    data = json.loads(path.read_text(encoding="utf-8"))
    # Never print the raw access token; show metadata only.
    safe = {k: v for k, v in data.items() if k not in {"access_token", "refresh_token"}}
    safe["access_token_preview"] = (data.get("access_token") or "")[:8] + "..."
    print(json.dumps(safe, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_auth = sub.add_parser(
        "authorize", help="Interactive OAuth flow — opens a URL, accepts a code."
    )
    p_auth.add_argument("--scope", default=None, help=f"OAuth scope (default {_DEFAULT_SCOPE})")
    p_auth.set_defaults(func=cmd_authorize)

    p_refresh = sub.add_parser(
        "refresh", help="Exchange the saved refresh_token for a new access_token."
    )
    p_refresh.set_defaults(func=cmd_refresh)

    p_show = sub.add_parser("show", help="Print token file metadata (without secrets).")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
