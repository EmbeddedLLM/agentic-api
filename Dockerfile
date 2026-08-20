# syntax=docker/dockerfile:1.7

ARG RUST_VERSION=1.96.0
ARG DEBIAN_VERSION=bookworm
ARG RUST_IMAGE_DIGEST=sha256:5e2214abe154fe26e39f64488952e5c991eeed1d6d6da7cc8381ae83927f0cfc
ARG DEBIAN_IMAGE_DIGEST=sha256:7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818

FROM rust:${RUST_VERSION}-${DEBIAN_VERSION}@${RUST_IMAGE_DIGEST} AS rust-build

ARG CARGO_BUILD_JOBS=4
ENV CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS}

WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,target=/workspace/target,sharing=locked \
    cargo build --locked --release -p agentic-server && \
    install -Dm755 -s target/release/agentic-server /out/agentic-server

FROM debian:${DEBIAN_VERSION}-slim@${DEBIAN_IMAGE_DIGEST} AS runtime

ARG RUNTIME_GID=0
ARG RUNTIME_UID=10001

RUN apt-get update && \
    apt-get install --yes --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /var/lib/agentic-api && \
    chown "${RUNTIME_UID}:${RUNTIME_GID}" /var/lib/agentic-api && \
    chmod g=u,g+s /var/lib/agentic-api

COPY --from=rust-build /out/agentic-server /usr/local/bin/agentic-server
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ARG OCI_CREATED=""
ARG OCI_BUILD_PIPELINE=local
ARG OCI_BUILD_URL=""
ARG OCI_REVISION=""
ARG OCI_SOURCE="https://github.com/vllm-project/agentic-api"
ARG OCI_VERSION=""

LABEL org.opencontainers.image.created="${OCI_CREATED}" \
      org.opencontainers.image.description="Rust gateway for stateful agentic APIs backed by vLLM" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${OCI_REVISION}" \
      org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.title="agentic-api" \
      org.opencontainers.image.url="${OCI_BUILD_URL}" \
      org.opencontainers.image.version="${OCI_VERSION}" \
      ai.vllm.build.commit="${OCI_REVISION}" \
      ai.vllm.build.pipeline="${OCI_BUILD_PIPELINE}" \
      ai.vllm.build.url="${OCI_BUILD_URL}" \
      ai.vllm.image.tag="${OCI_VERSION}"

WORKDIR /var/lib/agentic-api
USER ${RUNTIME_UID}:${RUNTIME_GID}

ENV GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=9000 \
    AGENTIC_API_HOME=/var/lib/agentic-api

EXPOSE 9000
ENTRYPOINT ["docker-entrypoint.sh"]
