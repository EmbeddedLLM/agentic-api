# Authenticate with GitHub through Dex

Use GitHub identities to authenticate callers by placing an OpenID Connect (OIDC) provider such as Dex in front of
agentic-api:

```text
GitHub OAuth → Dex OIDC ID token → agentic-api bearer validation
```

GitHub's API and OAuth access tokens are not OIDC ID tokens, so callers cannot send them directly to agentic-api.
Dex is the tested provider in this walkthrough, but any OIDC provider can be used when its issuer, audience,
asymmetric signing keys, and claims satisfy the gateway's
[validation contract](../design/oidc-bearer-authentication.md).

!!! warning "Security boundary"

    This configuration authenticates the caller at the request boundary. Tenant isolation and persisted-state
    authorization are separate work tracked by [issue #107](https://github.com/vllm-project/agentic-api/issues/107).
    agentic-api consumes the inbound ID token; it must not forward that token to the inference service.
    `OPENAI_API_KEY` is a separate upstream inference credential.

## Architecture and security boundary

GitHub authenticates the user, and Dex turns that result into a Dex-issued OIDC ID token for the client. agentic-api
validates that token at its `/v1/*` request boundary, then uses `OPENAI_API_KEY` independently when it calls the
inference service. The gateway performs OIDC discovery and loads the JSON Web Key Set (JWKS) before it begins
listening; see the [OIDC validation contract](../design/oidc-bearer-authentication.md) for issuer, key, claim, and
loopback-HTTP requirements.

## Local validation

This walkthrough is for local development only. Every HTTP URL below is loopback-only and development-only, and the
Dex `memory` storage setting is also development-only. Do not reuse these HTTP URLs or in-memory storage in a
production deployment.

### Prerequisites and local values

Install Docker, `jq`, `oauth2c`, and have an inference endpoint running at `http://127.0.0.1:5050`. Install
`oauth2c` with the [official package command for your platform](https://github.com/SecureAuthCorp/oauth2c#installation),
for example on macOS:

```bash
brew install cloudentity/tap/oauth2c
```

Use these development-only values. Each HTTP address in this table is loopback-only and development-only.

| Setting | Development value | Used by |
| --- | --- | --- |
| Dex issuer | `http://127.0.0.1:5556/dex` | Dex and `OIDC_ISSUER` |
| OIDC audience/client ID | `agentic-api-local` | Dex, `oauth2c`, and `OIDC_AUDIENCE` |
| OIDC client callback | `http://localhost:9876/callback` | Dex and `oauth2c` |
| GitHub OAuth callback | `http://127.0.0.1:5556/dex/callback` | GitHub and the Dex connector |

### Register the GitHub OAuth App

Create a [GitHub OAuth App](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/creating-an-oauth-app) with
these loopback-only, development-only URLs:

```text
Homepage URL: http://127.0.0.1:5556/dex
Authorization callback URL: http://127.0.0.1:5556/dex/callback
```

The GitHub callback is the Dex issuer plus `/callback`, not the `oauth2c` callback. Supply the generated client ID
and secret to Dex through environment variables or a secret manager. Never put the secret in the repository, image,
build arguments, command line, or shell history.

For a local shell, enter the values interactively so the secret is not echoed:

```bash
printf 'GitHub OAuth App client ID: '
IFS= read -r GITHUB_CLIENT_ID
printf 'GitHub OAuth App client secret: '
IFS= read -rs GITHUB_CLIENT_SECRET
printf '\n'
export GITHUB_CLIENT_ID GITHUB_CLIENT_SECRET
```

### Configure and run Dex

Create `dex.local.yaml` outside version control. This complete configuration is development-only: its HTTP issuer and
callback URLs are loopback-only, and `type: memory` loses all sessions and signing state when Dex restarts.

```yaml
issuer: http://127.0.0.1:5556/dex

storage:
  type: memory

web:
  http: 0.0.0.0:5556

connectors:
  - type: github
    id: github
    name: GitHub
    config:
      clientID: $GITHUB_CLIENT_ID
      clientSecret: $GITHUB_CLIENT_SECRET
      redirectURI: http://127.0.0.1:5556/dex/callback

staticClients:
  - id: agentic-api-local
    name: agentic-api local
    public: true
    redirectURIs:
      - http://localhost:9876/callback
```

Dex expands the two environment references at runtime. `public: true` enables a public client using authorization
code plus Proof Key for Code Exchange (PKCE), so the caller has no client secret. The in-memory store is unsuitable
for production because it loses sessions and signing state on restart. Dex listens on `0.0.0.0` only inside its
container; the Docker `--publish` setting below exposes it only at the loopback-only development address.

Run the pinned Dex image with the configuration mounted read-only:

```bash
docker run --rm --name agentic-api-dex \
  --publish 127.0.0.1:5556:5556 \
  --env GITHUB_CLIENT_ID \
  --env GITHUB_CLIENT_SECRET \
  --volume "$PWD/dex.local.yaml:/etc/dex/config.yaml:ro" \
  ghcr.io/dexidp/dex:v2.45.1 dex serve /etc/dex/config.yaml
```

See the [Dex GitHub connector documentation](https://dexidp.io/docs/connectors/github/) for connector options,
including organization and team restrictions.

### Run agentic-api

Supply the service credential independently of OIDC, then start the gateway with the matching loopback-only,
development-only issuer and audience:

```bash
export OPENAI_API_KEY
cargo run -p agentic-server -- \
  --llm-api-base http://127.0.0.1:5050 \
  --oidc-issuer http://127.0.0.1:5556/dex \
  --oidc-audience agentic-api-local
```

The gateway performs OIDC discovery and JWKS loading before it begins listening. `OPENAI_API_KEY` remains the
inference-service credential; it is not an OIDC caller credential.

### Sign in with PKCE

Obtain an ID token with the public client. `oauth2c` opens the GitHub login in a browser and listens only for its own
loopback-only, development-only callback. Put only the ID token in `OIDC_TOKEN`:

```bash
OIDC_TOKEN="$(
  oauth2c http://127.0.0.1:5556/dex \
    --client-id agentic-api-local \
    --response-types code \
    --response-mode query \
    --grant-type authorization_code \
    --auth-method none \
    --scopes openid,email,profile \
    --redirect-url http://localhost:9876/callback \
    --pkce \
    --silent |
    jq -er '.id_token'
)"
```

Do not print, log, paste, or commit the token, complete token response, authorization code, or PKCE verifier. Do
not decode claims in this walkthrough.

### Verify the request boundary

First verify the unauthenticated boundary:

```bash
curl -i http://127.0.0.1:9000/v1/models
```

Expected: `401 Unauthorized` with `WWW-Authenticate: Bearer`.

Then verify the GitHub-authenticated request:

```bash
curl --fail-with-body \
  --header "Authorization: Bearer $OIDC_TOKEN" \
  http://127.0.0.1:9000/v1/models
```

Expected: an upstream model-list response rather than an authentication error.

To verify credential separation, inspect an inference-service access log or use a mock inference service: the
upstream must receive `OPENAI_API_KEY`, never `$OIDC_TOKEN`. Do not print either credential.

## Codex

Create a user-owned executable helper outside the repository, for example `print-oidc-token`. It must send only the
ID token to stdout:

```sh
#!/usr/bin/env sh
set -eu

oauth2c http://127.0.0.1:5556/dex \
  --client-id agentic-api-local \
  --response-types code \
  --response-mode query \
  --grant-type authorization_code \
  --auth-method none \
  --scopes openid,email,profile \
  --redirect-url http://localhost:9876/callback \
  --pkce \
  --silent |
  jq -er '.id_token'
```

Use the helper with Codex's supported command-backed bearer authentication:

```toml
[model_providers.agentic-api.auth]
command = "/absolute/path/to/print-oidc-token"
args = []
refresh_interval_ms = 300000
```

This development helper initiates an interactive browser flow. Production operators should use their provider's
secure refresh or credential-helper workflow and keep refresh tokens out of repository files and logs.

## Claude Code

Use the token already obtained in the current shell:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:9000
export ANTHROPIC_AUTH_TOKEN="$OIDC_TOKEN"
unset ANTHROPIC_API_KEY

claude -p "summarize the files in this directory"
```

`ANTHROPIC_AUTH_TOKEN` must be refreshed before expiry and must not also be supplied as `ANTHROPIC_API_KEY`.

## Production checklist

- Use an HTTPS Dex issuer and HTTPS callbacks.
- Use a persistent, backed-up Dex datastore appropriate to the deployment topology.
- Inject secrets from a secret manager at runtime, never through image layers, build arguments, checked-in files, or
  shell history.
- Rotate the GitHub OAuth App secret.
- Use a stable, deployment-specific audience shared only by intended callers.
- Apply optional Dex `orgs` and `teams` restrictions when deployment policy requires them.
- Apply network and ingress controls around Dex and agentic-api.

## Troubleshooting

- **GitHub rejects the callback:** the GitHub OAuth App callback is the Dex issuer plus `/callback`. See the
  [Dex GitHub connector documentation](https://dexidp.io/docs/connectors/github/).
- **Discovery or startup fails:** Dex discovery's `issuer` must exactly equal `OIDC_ISSUER`, including scheme, host,
  port, and path. See the [OIDC validation contract](../design/oidc-bearer-authentication.md).
- **Token is rejected:** every ID-token `aud` value must equal `OIDC_AUDIENCE`, and any present `azp` must also equal
  it; a GitHub access token is not a substitute. See the
  [OIDC validation contract](../design/oidc-bearer-authentication.md).
- **HTTP issuer is rejected:** HTTP is allowed only for literal loopback issuers; use HTTPS elsewhere. See the
  [OIDC validation contract](../design/oidc-bearer-authentication.md).
- **Upstream sees the identity token:** configure `OPENAI_API_KEY` separately and stop forwarding the identity token.
  See the [OIDC validation contract](../design/oidc-bearer-authentication.md).
