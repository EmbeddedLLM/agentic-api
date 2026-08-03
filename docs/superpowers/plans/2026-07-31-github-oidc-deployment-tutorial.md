# GitHub OIDC Deployment Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a secret-free, copy-pasteable deployment tutorial for authenticating agentic-api callers with
GitHub identities through Dex-issued OpenID Connect (OIDC) ID tokens.

**Architecture:** Add one focused deployment page that explains the generic OIDC contract and then walks through the
validated GitHub OAuth → Dex OIDC → agentic-api flow. Wire that page into the existing navigation and authentication
entry points without changing runtime code or expanding into tenant authorization.

**Tech Stack:** MkDocs Material, Markdown, Docker, Dex v2.45.1, GitHub OAuth Apps, `oauth2c`, `jq`, Codex, Claude Code

## Global Constraints

- The tutorial lives at `docs/deploying/github-oidc.md` and appears as **Deploying → GitHub authentication with Dex**.
- Explain the generic OIDC contract before using Dex as the concrete, tested broker.
- Use the flow `GitHub OAuth → Dex OIDC ID token → agentic-api bearer validation`.
- State that GitHub API and OAuth access tokens are not OIDC ID tokens and are not accepted directly.
- State that authentication establishes a principal at the request boundary; tenant isolation remains issue #107.
- Use HTTP loopback and Dex in-memory storage only in a development-only example.
- Require HTTPS, persistent backed-up storage, runtime secret injection, secret rotation, a stable audience, and normal
  ingress controls for production.
- Do not include any generated client ID, client secret, token, authorization code, PKCE verifier, personal identity
  claim, machine-specific temporary path, or local database content.
- Keep `OPENAI_API_KEY` documented as the upstream inference-service credential, separate from the inbound OIDC ID
  token.
- Do not add a gateway browser callback, repository-owned token helper, Dex-only requirement, complete highly
  available Dex deployment, or tenant authorization.
- Follow the preferred prose in `TERMINOLOGY.md` and preserve exact protocol field names.
- Keep Markdown lines within the repository's 120-character formatting convention.

---

## File Map

- Create `docs/deploying/github-oidc.md`: owns the complete GitHub-backed OIDC deployment walkthrough, development
  example, client setup, production guidance, and troubleshooting.
- Modify `mkdocs.yaml`: adds the new tutorial to the Deploying navigation.
- Modify `docs/deploying/container.md`: routes operators from the container OIDC settings to the runnable tutorial.
- Modify `docs/api/index.md`: routes API readers from the authentication contract to the runnable tutorial.
- Modify `README.md`: routes Codex and Claude Code users from their OIDC snippets to the complete setup.
- Preserve `docs/superpowers/specs/2026-07-31-github-oidc-deployment-tutorial-design.md`: this is the approved design
  record and is not implementation content.

### Task 1: Author the GitHub and Dex deployment tutorial

**Files:**

- Create: `docs/deploying/github-oidc.md`
- Reference: `docs/design/oidc-bearer-authentication.md`
- Reference: `docs/deploying/container.md`
- Reference: `TERMINOLOGY.md`

**Interfaces:**

- Consumes: the existing `OIDC_ISSUER`, `OIDC_AUDIENCE`, and `OPENAI_API_KEY` configuration contract.
- Produces: a page with stable sections for architecture, local validation, Codex, Claude Code, production, and
  troubleshooting that the navigation and cross-links in Task 2 target.

- [ ] **Step 1: Confirm the runtime contract and external command syntax**

Read the local OIDC contract and the official sources used by the walkthrough:

```bash
sed -n '1,260p' docs/design/oidc-bearer-authentication.md
```

Use these primary sources for claims about external tools:

- `https://dexidp.io/docs/connectors/github/`
- `https://dexidp.io/docs/getting-started/`
- `https://github.com/SecureAuthCorp/oauth2c`
- `https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/creating-an-oauth-app`

Confirm these exact `oauth2c` flags remain supported before publishing the command:

```bash
oauth2c --help |
  rg -- '--auth-method|--client-id|--grant-type|--pkce|--redirect-url'
oauth2c --help |
  rg -- '--response-mode|--response-types|--scopes|--silent'
```

Expected: every flag is listed; `--redirect-url` documents `http://localhost:9876/callback` as the default.

- [ ] **Step 2: Write the architecture and security-boundary introduction**

Create `docs/deploying/github-oidc.md` with this opening structure and meaning:

```markdown
# Authenticate with GitHub through Dex

Use GitHub identities to authenticate callers by placing an OpenID Connect (OIDC) provider such as Dex in front of
agentic-api:

```text
GitHub OAuth → Dex OIDC ID token → agentic-api bearer validation
```

GitHub's API and OAuth access tokens are not OIDC ID tokens, so callers cannot send them directly to agentic-api.
Dex is the tested provider in this walkthrough, but any OIDC provider can be used when its issuer, audience,
asymmetric signing keys, and claims satisfy the gateway's validation contract.
```

Immediately follow the introduction with an admonition that says:

- this authenticates the caller at the request boundary;
- tenant isolation and persisted-state authorization are separate work tracked by issue #107;
- the inbound ID token is consumed by agentic-api and must not be forwarded to the inference service; and
- `OPENAI_API_KEY` is the separate upstream inference credential.

Link “validation contract” to `../design/oidc-bearer-authentication.md` and issue #107 to
`https://github.com/vllm-project/agentic-api/issues/107`.

- [ ] **Step 3: Add prerequisites and choose the three local values**

Document Docker, a running inference endpoint, `oauth2c`, and `jq` as prerequisites. Install `oauth2c` using the
officially documented package command for the reader's platform rather than a repository script.

Use this development-only value table:

| Setting | Development value | Used by |
| --- | --- | --- |
| Dex issuer | `http://127.0.0.1:5556/dex` | Dex and `OIDC_ISSUER` |
| OIDC audience/client ID | `agentic-api-local` | Dex, `oauth2c`, and `OIDC_AUDIENCE` |
| OIDC client callback | `http://localhost:9876/callback` | Dex and `oauth2c` |
| GitHub OAuth callback | `http://127.0.0.1:5556/dex/callback` | GitHub and the Dex connector |

Label every HTTP address and in-memory setting in this section as loopback-only and development-only.

- [ ] **Step 4: Document GitHub OAuth App registration without embedding credentials**

Tell the operator to create a GitHub OAuth App with:

```text
Homepage URL: http://127.0.0.1:5556/dex
Authorization callback URL: http://127.0.0.1:5556/dex/callback
```

Explain that the GitHub callback is the Dex issuer plus `/callback`, not the `oauth2c` callback. Tell the operator to
provide the generated client ID and secret to Dex through the environment or a secret manager and never place the
secret in the repository, image, build arguments, command line, or shell history.

For the local shell, show interactive environment setup that does not echo the secret:

```bash
printf 'GitHub OAuth App client ID: '
IFS= read -r GITHUB_CLIENT_ID
printf 'GitHub OAuth App client secret: '
IFS= read -rs GITHUB_CLIENT_SECRET
printf '\n'
export GITHUB_CLIENT_ID GITHUB_CLIENT_SECRET
```

- [ ] **Step 5: Add the complete development-only Dex configuration**

Provide a `dex.local.yaml` example containing no literal secret:

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

Explain that Dex expands the two environment references at runtime and that `public: true` enables a public client
using authorization code plus Proof Key for Code Exchange (PKCE), so there is no client secret in the caller.

Run the pinned image with the config mounted read-only:

```bash
docker run --rm --name agentic-api-dex \
  --publish 127.0.0.1:5556:5556 \
  --env GITHUB_CLIENT_ID \
  --env GITHUB_CLIENT_SECRET \
  --volume "$PWD/dex.local.yaml:/etc/dex/config.yaml:ro" \
  ghcr.io/dexidp/dex:v2.45.1 dex serve /etc/dex/config.yaml
```

State that the in-memory store loses sessions and signing state on restart and is unsuitable for production.

- [ ] **Step 6: Document matching agentic-api configuration and credential separation**

Show the gateway process with a service credential supplied independently of OIDC:

```bash
export OPENAI_API_KEY
cargo run -p agentic-server -- \
  --llm-api-base http://127.0.0.1:5050 \
  --oidc-issuer http://127.0.0.1:5556/dex \
  --oidc-audience agentic-api-local
```

Before finalizing the page, run `cargo run -p agentic-server -- --help` and confirm the three option names shown above
match the current binary. Explain that gateway startup performs OIDC discovery and JWKS loading before it begins
listening.

- [ ] **Step 7: Add the PKCE login and token handling commands**

Obtain an ID token with a public client and only place the ID token in the `OIDC_TOKEN` variable:

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

Explain that `oauth2c` opens the GitHub login in a browser and listens only for its own callback. Warn readers not to
print, log, paste, or commit the token, complete token response, authorization code, or PKCE verifier. Do not decode
claims in the tutorial.

- [ ] **Step 8: Add the three verification checks**

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

Finally explain how to verify credential separation with an inference-service access log or mock: the upstream must
receive `OPENAI_API_KEY`, never `$OIDC_TOKEN`. Do not suggest printing either credential.

- [ ] **Step 9: Add Codex and Claude Code client sections**

For Codex, create a user-owned executable helper outside the repository. The helper must send only the ID token to
stdout:

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

Use that helper with the current README's supported configuration:

```toml
[model_providers.agentic-api.auth]
command = "/absolute/path/to/print-oidc-token"
args = []
refresh_interval_ms = 300000
```

Explain that this development helper initiates an interactive browser flow; production operators should use their
provider's secure refresh or credential-helper workflow and keep refresh tokens out of repository files and logs.

For Claude Code, use the token already obtained in the current shell:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:9000
export ANTHROPIC_AUTH_TOKEN="$OIDC_TOKEN"
unset ANTHROPIC_API_KEY

claude -p "summarize the files in this directory"
```

Explain that `ANTHROPIC_AUTH_TOKEN` must be refreshed before expiry and must not also be supplied as
`ANTHROPIC_API_KEY`.

- [ ] **Step 10: Add production guidance and troubleshooting**

Add a production checklist requiring:

- an HTTPS Dex issuer and HTTPS callbacks;
- a persistent, backed-up Dex datastore appropriate to the deployment topology;
- secret-manager injection at runtime, never image layers, build arguments, checked-in files, or shell history;
- rotation of the GitHub OAuth App secret;
- a stable, deployment-specific audience shared only by intended callers;
- optional Dex `orgs` and `teams` restrictions when deployment policy requires them; and
- network and ingress controls around Dex and agentic-api.

Add a troubleshooting section with these exact diagnoses:

- **GitHub rejects the callback:** the GitHub OAuth App callback is the Dex issuer plus `/callback`.
- **Discovery or startup fails:** Dex discovery's `issuer` must exactly equal `OIDC_ISSUER`, including scheme, host,
  port, and path.
- **Token is rejected:** the ID token `aud` must include `OIDC_AUDIENCE`; a GitHub access token is not a substitute.
- **HTTP issuer is rejected:** HTTP is allowed only for literal loopback issuers; use HTTPS elsewhere.
- **Upstream sees the identity token:** configure `OPENAI_API_KEY` separately and stop forwarding the identity token.

Link the relevant rows to the local OIDC design and the official Dex GitHub connector documentation.

- [ ] **Step 11: Run focused content and privacy checks**

Run:

```bash
rg -n '^## ' docs/deploying/github-oidc.md
rg -n 'GitHub OAuth|Dex OIDC ID token|OIDC_ISSUER|OIDC_AUDIENCE|OPENAI_API_KEY|issue #107|PKCE' \
  docs/deploying/github-oidc.md
rg -n '(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)' \
  docs/deploying/github-oidc.md
git diff --check
```

Expected: all required concepts appear; the credential/JWT scan prints no matches; `git diff --check` exits zero.
Manually inspect the diff for personal names, usernames, email addresses, subject claims, generated IDs, local database
paths, and tokens. Remove any such value before committing.

- [ ] **Step 12: Commit the standalone tutorial**

```bash
git add docs/deploying/github-oidc.md
git commit -s -m "docs: add GitHub OIDC deployment tutorial"
```

### Task 2: Integrate the tutorial into the documentation entry points

**Files:**

- Modify: `mkdocs.yaml:92-101`
- Modify: `docs/deploying/container.md:70-85`
- Modify: `docs/api/index.md:1-35`
- Modify: `README.md:127-174`

**Interfaces:**

- Consumes: `docs/deploying/github-oidc.md` from Task 1.
- Produces: discoverable navigation and cross-links from every OIDC configuration surface named in the approved spec.

- [ ] **Step 1: Add the MkDocs navigation entry**

Change the Deploying navigation to:

```yaml
  - Deploying:
    - Container: deploying/container.md
    - GitHub authentication with Dex: deploying/github-oidc.md
```

- [ ] **Step 2: Link the container guide to the runnable tutorial**

After the existing OIDC validation-contract link in `docs/deploying/container.md`, add a sentence with this intent:

```markdown
For a runnable GitHub-backed setup, follow [GitHub authentication with Dex](github-oidc.md).
```

Keep the design link for protocol behavior and the tutorial link for deployment steps.

- [ ] **Step 3: Link the API authentication section to the runnable tutorial**

After the validation-contract link in `docs/api/index.md`, add:

```markdown
For a complete GitHub-backed deployment example, see
[GitHub authentication with Dex](../deploying/github-oidc.md).
```

- [ ] **Step 4: Link both README client examples to the tutorial**

After the Codex OIDC credential-helper paragraph, add:

```markdown
See [GitHub authentication with Dex](docs/deploying/github-oidc.md) for a complete GitHub login, token-helper, and
gateway setup.
```

After the Claude Code bearer-token paragraph, add:

```markdown
The same [GitHub authentication with Dex](docs/deploying/github-oidc.md) guide shows how to obtain the ID token
without embedding a client secret in Claude Code.
```

- [ ] **Step 5: Build the documentation and inspect navigation/link output**

```bash
uv run --with-requirements docs/requirements.txt mkdocs build
```

Expected: exit zero. Review every warning and verify no new warning points to `deploying/github-oidc.md` or a link
added by this task. Pre-existing unrelated warnings may remain but must be identified in the final test report.

- [ ] **Step 6: Run repository documentation hooks and diff checks**

```bash
uvx pre-commit run --all-files
git diff --check
git diff -- README.md mkdocs.yaml docs/api/index.md docs/deploying/container.md docs/deploying/github-oidc.md
```

Expected: pre-commit and `git diff --check` pass. Review the rendered-link destinations, exact navigation label,
credential separation language, and absence of real secrets or personal test data.

- [ ] **Step 7: Commit navigation and cross-links**

```bash
git add README.md mkdocs.yaml docs/api/index.md docs/deploying/container.md
git commit -s -m "docs: link GitHub OIDC deployment guide"
```

### Task 3: Run the required review gates and update PR #149

**Files:**

- Review: every file changed since `origin/main`
- Modify only when feedback is actionable and within issue #102.
- Update: existing GitHub PR `https://github.com/vllm-project/agentic-api/pull/149`

**Interfaces:**

- Consumes: the tutorial and documentation integration from Tasks 1 and 2.
- Produces: a reviewed, verified, secret-free update to the existing OIDC pull request; do not open a second PR.

- [ ] **Step 1: Apply the Rust guidance during self-review**

Read the `rust-skills` instructions from the available skills catalog completely. Although this slice is
documentation-only, confirm the tutorial matches the implemented Rust configuration, authentication boundary, error
behavior, and credential forwarding behavior. Do not change Rust unless the documentation exposes an in-scope
correctness defect.

- [ ] **Step 2: Run the read-only Claude review**

Read the `claude-review` instructions from the available skills catalog completely and run its worktree review against
the diff from `origin/main`. Record every finding and classify it as actionable, already addressed, out of scope, or
incorrect with specific evidence.

- [ ] **Step 3: Run the gstack pre-landing review**

Read the `review` instructions from the available skills catalog completely and run the prescribed pre-landing review
against the same diff. Pay particular attention to credential leakage, misleading production advice, broken commands,
broken links, and scope expansion into issue #107.

- [ ] **Step 4: Resolve all actionable review feedback**

For each actionable finding:

1. Reproduce or verify it against the documentation and current binary.
2. Make the smallest in-scope correction.
3. Re-run the focused command, link, privacy, or docs-build check that proves the correction.
4. Re-run the relevant review until it reports no unresolved actionable findings.

If fixes change files, commit them separately:

```bash
git add README.md mkdocs.yaml docs/api/index.md docs/deploying/container.md docs/deploying/github-oidc.md
git commit -s -m "docs: address OIDC tutorial review"
```

Do not broaden the PR into tenant authorization, a token-helper implementation, or a production Dex platform.

- [ ] **Step 5: Run final local verification**

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
uv run --with-requirements docs/requirements.txt mkdocs build
uvx pre-commit run --all-files
git diff --check origin/main...HEAD
git status --short
```

Expected: all commands pass and `git status --short` is empty. Review all MkDocs warnings and confirm none were
introduced by this tutorial.

Run the final secret/privacy scan over the complete OIDC branch diff:

```bash
git diff origin/main...HEAD | rg -n \
  '(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)'
```

Expected: no output. Manually confirm that the diff contains no real client ID, secret, token, authorization code,
PKCE verifier, personal identity claim, machine-specific temporary path, or local database content.

- [ ] **Step 6: Push the reviewed commits and update the existing PR description**

```bash
git push origin codex/issue-102-oidc-auth
```

Update PR #149 rather than creating another PR. Preserve its existing OIDC implementation summary, then add:

```markdown
- documents a tested GitHub → Dex → agentic-api deployment flow, including Codex and Claude Code bearer setup
```

Add these exact categories to its **Test Plan** with the real final results:

```markdown
- live local GitHub OAuth login through Dex v2.45.1 using authorization code with PKCE
- unauthenticated and authenticated `/v1/models` checks, plus authenticated `/v1/responses/compact`
- upstream credential-separation check proving the OIDC ID token was not forwarded
- `uv run --with-requirements docs/requirements.txt mkdocs build`
- `uvx pre-commit run --all-files`
```

Do not include any token, generated identifier, user claim, OAuth App secret, or local temporary path in the PR body.

- [ ] **Step 7: Verify GitHub state and report the outcome**

```bash
gh pr view 149 --json url,state,isDraft,mergeable,headRefName,baseRefName,statusCheckRollup
```

Expected: PR #149 targets `main`, uses `codex/issue-102-oidc-auth`, and contains the pushed documentation commits.
Wait for required checks, investigate failures caused by this branch, and report:

- what documentation changed;
- the live GitHub/Dex checks already completed;
- all final local checks;
- Claude and gstack findings and how each actionable item was resolved;
- the overlap audit result for issue #102; and
- the PR URL.
