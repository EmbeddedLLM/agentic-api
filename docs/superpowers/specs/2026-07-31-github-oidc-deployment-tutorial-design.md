# GitHub OIDC deployment tutorial design

## Goal

Add a user-facing deployment tutorial that explains how to authenticate agentic-api callers with GitHub identities
through an OpenID Connect (OIDC) broker. The guide must be runnable, distinguish GitHub OAuth from OIDC, and never
embed real credentials or personal identity data.

## Audience and placement

The primary audience is an operator deploying agentic-api who wants GitHub-backed user authentication. The tutorial
will live at `docs/deploying/github-oidc.md` and appear in the MkDocs navigation as **Deploying → GitHub authentication
with Dex**.

The page will be linked from:

- the OIDC section of `docs/deploying/container.md`;
- the authentication section of `docs/api/index.md`; and
- the Codex and Claude Code authentication examples in `README.md`.

## Reader model

The tutorial will establish this data flow before presenting commands:

```text
GitHub OAuth → Dex OIDC ID token → agentic-api bearer validation
```

It will explain that GitHub API and OAuth access tokens are not OIDC ID tokens and therefore cannot be passed directly
to agentic-api. Dex is the concrete, tested broker in the walkthrough, but agentic-api accepts tokens from any OIDC
provider that satisfies its documented issuer, audience, asymmetric-signature, and claim requirements.

The guide will also state that this feature authenticates a principal at the request boundary. Tenant isolation and
persisted-state authorization remain separate work under issue #107.

## Tutorial structure

The tutorial will lead the reader through these steps:

1. Choose the Dex issuer, agentic-api audience, and OIDC client callback URL.
2. Register a GitHub OAuth App whose authorization callback is exactly the Dex issuer plus `/callback`.
3. Configure the Dex GitHub connector with the OAuth App client ID and client secret supplied through environment
   variables or a secret manager.
4. Register a public Dex client that uses the authorization-code flow with Proof Key for Code Exchange (PKCE).
5. Start agentic-api with matching `OIDC_ISSUER` and `OIDC_AUDIENCE` values while keeping `OPENAI_API_KEY` as the
   separate inference-service credential.
6. Obtain an OIDC ID token with `oauth2c`, emitting only the `id_token` when used as a credential helper.
7. Configure Codex command-backed bearer authentication or Claude Code's `ANTHROPIC_AUTH_TOKEN`.
8. Verify that an unauthenticated `/v1/*` request returns `401`, the GitHub-authenticated request succeeds, and the
   identity token is not forwarded to the inference service.

## Development and production guidance

The page will include a runnable loopback Docker example for validation. That example may use HTTP loopback addresses
and Dex's in-memory storage, and it will be labelled as development-only.

The production section will require:

- an HTTPS Dex issuer and HTTPS callback URL;
- persistent, backed-up Dex storage appropriate to the deployment topology;
- runtime secret injection instead of image layers, build arguments, checked-in files, or shell history;
- GitHub OAuth App secret rotation;
- a stable, deployment-specific audience;
- optional GitHub organization or team restrictions when the deployment needs them; and
- normal network and ingress controls around both Dex and agentic-api.

## Secret and privacy boundaries

Every example will use placeholders or environment-variable references. The tutorial will not contain:

- the temporary OAuth App ID or client ID used during validation;
- a GitHub client secret, Dex token, refresh token, authorization code, or PKCE verifier;
- personal names, usernames, email addresses, subject identifiers, or other token claims; or
- machine-specific temporary paths and local database contents.

The guide will explicitly warn readers not to print tokens in shared logs and not to confuse the inbound identity token
with `OPENAI_API_KEY`, which is an upstream service credential.

## Troubleshooting

Troubleshooting will cover the failure modes observed or validated during the live test and implementation review:

- the GitHub OAuth App callback must be the Dex issuer plus `/callback`;
- Dex discovery issuer and agentic-api `OIDC_ISSUER` must match exactly;
- the ID token audience must equal `OIDC_AUDIENCE`;
- GitHub API or OAuth access tokens are rejected because they are not Dex-issued OIDC ID tokens;
- HTTP issuers are accepted only for literal loopback addresses; and
- an OIDC identity token must be consumed by the gateway rather than forwarded upstream.

## Documentation verification

The implementation will be verified with:

- a placeholder and secret-pattern scan over the new and modified documentation;
- a link and navigation review;
- `uv run --with-requirements docs/requirements.txt mkdocs build`;
- `uvx pre-commit run --all-files`; and
- a final diff review confirming that no generated credentials or personal test data were committed.

## Out of scope

This documentation change will not add a browser callback to agentic-api, ship or support a repository-owned token
helper, prescribe Dex as the only supported broker, document a complete highly available Dex deployment, or implement
tenant authorization from issue #107.
