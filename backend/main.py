from app_factory import create_app
from routes.http import router as http_router
from routes.ws import router as ws_router
import os

app = create_app()
app.include_router(http_router)
app.include_router(ws_router)


if __name__ == "__main__":
    import uvicorn

    ws_ping_interval = float(os.environ.get("MEDGEMMA_WS_PING_INTERVAL_S", "60"))
    ws_ping_timeout = float(os.environ.get("MEDGEMMA_WS_PING_TIMEOUT_S", "120"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=ws_ping_interval,
        ws_ping_timeout=ws_ping_timeout,
    )
