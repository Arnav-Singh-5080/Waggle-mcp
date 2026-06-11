from __future__ import annotations

from .provider import ExampleOIDCProvider

__version__ = "0.0.1"

WAGGLE_PLUS_CAPABILITIES = ("oidc_sso", "rbac")


def build_identity_provider() -> ExampleOIDCProvider:
    return ExampleOIDCProvider()
