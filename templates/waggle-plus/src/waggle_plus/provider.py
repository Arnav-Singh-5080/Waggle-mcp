from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode


@dataclass(slots=True)
class ExampleOIDCProvider:
    issuer: str = "https://id.example.com"
    client_id: str = "waggle-plus-client"

    def status(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "client_id": self.client_id,
            "supported_roles": ["Owner", "Admin", "Developer", "Viewer", "Auditor"],
            "supported_permissions": [
                "read:data",
                "write:data",
                "delete:data",
                "export:data",
                "manage:users",
                "manage:security",
            ],
        }

    def authorize_url(self, *, redirect_uri: str, state: str) -> str:
        query = urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": "openid profile email",
                "state": state,
            }
        )
        return f"{self.issuer}/authorize?{query}"

    def exchange_code(self, *, code: str, redirect_uri: str, state: str) -> dict[str, Any]:
        return {
            "subject": f"user:{code}",
            "issuer": self.issuer,
            "email": "admin@example.com",
            "display_name": "Admin User",
            "roles": ["Owner", "Admin"],
            "expires_at": "2026-12-31T23:59:59Z",
            "redirect_uri_seen": redirect_uri,
            "state_seen": state,
        }

    def map_roles(self, *, session: dict[str, Any]) -> dict[str, Any]:
        source_roles = [str(value) for value in session.get("roles", [])]
        primary_role = "Admin" if "Admin" in source_roles else "Viewer"
        permissions = (
            ["read:data", "write:data", "manage:users", "manage:security"]
            if primary_role == "Admin"
            else ["read:data"]
        )
        return {
            "primary_role": primary_role,
            "roles": source_roles,
            "permissions": permissions,
            "source_claims": {"roles": source_roles},
        }

    def check_permission(
        self,
        *,
        resolved: dict[str, Any],
        permission: str,
        resource_type: str = "",
        resource_id: str = "",
    ) -> dict[str, Any]:
        matched_permissions = [value for value in resolved.get("permissions", []) if str(value) == permission]
        allowed = bool(matched_permissions)
        return {
            "allowed": allowed,
            "reason": "matched_permission" if allowed else "permission_missing",
            "matched_roles": [str(value) for value in resolved.get("roles", [])],
            "matched_permissions": matched_permissions,
            "resource_type": resource_type,
            "resource_id": resource_id,
        }
