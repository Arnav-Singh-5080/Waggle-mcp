from __future__ import annotations

from waggle_plus import WAGGLE_PLUS_CAPABILITIES, build_identity_provider


def test_capabilities_advertise_identity_and_rbac() -> None:
    assert "oidc_sso" in WAGGLE_PLUS_CAPABILITIES
    assert "rbac" in WAGGLE_PLUS_CAPABILITIES


def test_provider_implements_public_core_contract() -> None:
    provider = build_identity_provider()

    status = provider.status()
    assert isinstance(status, dict)
    assert "issuer" in status
    assert "supported_roles" in status

    authorize_url = provider.authorize_url(
        redirect_uri="https://waggle.example.com/callback",
        state="state-123",
    )
    assert "redirect_uri=" in authorize_url
    assert "state=state-123" in authorize_url

    session = provider.exchange_code(
        code="oidc-code-123",
        redirect_uri="https://waggle.example.com/callback",
        state="state-123",
    )
    assert session["subject"]

    resolved = provider.map_roles(session=session)
    assert resolved["primary_role"]

    decision = provider.check_permission(
        resolved=resolved,
        permission="read:data",
        resource_type="tenant",
        resource_id="workspace-a",
    )
    assert "allowed" in decision
