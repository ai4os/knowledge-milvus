import jwt

KEYCLOAK_URL = "https://login.cloud.ai4eosc.eu/realms/ai4eosc"
jwk_client = jwt.PyJWKClient(f"{KEYCLOAK_URL}/protocol/openid-connect/certs")


def get_user_info(token: str) -> dict:
    # Fetch the public key from Keycloak's JWKS endpoint
    signing_key = jwk_client.get_signing_key_from_jwt(token).key

    # Decode and verify token signature and issuer
    payload = jwt.decode(
        token,
        signing_key,
        issuer=KEYCLOAK_URL,
        algorithms=["RS256"],
        options={
            "verify_signature": True,
            "verify_iss": True,
            "verify_exp": True,
            "verify_aud": False,
        },
    )

    return {
        "id": payload.get("sub"),
    }