# vLLM Agentic API

Agentic API is a Rust-native gateway for OpenAI-compatible inference servers. It provides the Responses API with
conversation state, tool execution, streaming, and continuation. Point it at an existing vLLM server to use open
models with your applications and coding clients.

The `agentic-api` Python package includes the Rust gateway, the `agentic` CLI, and a Python launcher. The base
installation does not install vLLM or GPU dependencies.

## Install

Requires Python 3.10 or newer. Prebuilt wheels are available for Linux x86_64 (glibc 2.17 or newer), macOS Intel,
and macOS Apple Silicon.

```bash
python -m pip install agentic-api
python -m agentic_api --version
```

## Start the server

With an inference server already running at `http://127.0.0.1:8000`:

```bash
python -m agentic_api serve --vllm-base-url http://127.0.0.1:8000
```

The gateway listens on port 9000 by default. Send Responses API requests to `http://127.0.0.1:9000/v1/responses`.

To install the pinned vLLM runtime and let the launcher manage a local inference process on a supported Linux host:

```bash
python -m pip install 'agentic-api[local]'
python -m agentic_api serve --model MODEL_ID
```

Run `python -m agentic_api doctor --mode remote --json` to check your installation.

## Learn more

- [Website and quickstart](https://vllm-project.github.io/agentic-api/)
- [Documentation](https://vllm-project.github.io/agentic-api/docs/latest)
- [Source code and contributing](https://github.com/vllm-project/agentic-api)
- [Report an issue](https://github.com/vllm-project/agentic-api/issues)
