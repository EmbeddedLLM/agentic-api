from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.core.proxy import ProxyClientManager
from agentic_api.database.schema import SchemaManager
from agentic_api.database.db_engine import create_db_engine_async
from agentic_api.routers import responses
from agentic_api.store.conversation import ConversationStore
from agentic_api.store.response import ResponseStore


def create_app(runtime_config: RuntimeConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime_config = runtime_config
        app.state.proxy_client_manager = ProxyClientManager()

        if runtime_config.response_store_enabled:
            engine = create_db_engine_async(
                db_url=runtime_config.db_url,
                db_dialect=runtime_config.db_dialect,
            )
            schema_manager = SchemaManager(engine)
            await schema_manager.ensure_ready(
                gateway_workers=runtime_config.gateway_workers,
                db_dialect=runtime_config.db_dialect,
            )
            app.state.response_store = ResponseStore(engine=engine)
            app.state.conversation_store = ConversationStore(
                engine=engine, db_dialect=runtime_config.db_dialect
            )
        else:
            app.state.conversation_store = None
            app.state.response_store = None

        yield

        await app.state.proxy_client_manager.aclose()
        if runtime_config.response_store_enabled:
            await engine.dispose()

    app = FastAPI(
        title="Agentic API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(responses.router)
    return app
