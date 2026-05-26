import logging
import os

import uvicorn
from dotenv import load_dotenv

from proxy.config import ProxyConfig
from proxy.server import create_app
from proxy.tracker import RequestTracker

_HERE = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(_HERE, ".env"))


def main() -> None:
    config = ProxyConfig.load(os.path.join(_HERE, "config.yaml"))

    # Allow env vars to override config.yaml server settings
    host = os.getenv("PROXY_HOST", config.server.host)
    port = int(os.getenv("PROXY_PORT", str(config.server.port)))

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    db_path = config.tracking.db
    if not os.path.isabs(db_path):
        db_path = os.path.join(_HERE, db_path)
    tracker = RequestTracker(db_path) if config.tracking.enabled else None
    app = create_app(config, tracker)

    logging.getLogger(__name__).info(f"thrift-flow listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
