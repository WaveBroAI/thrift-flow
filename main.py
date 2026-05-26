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

    logging_level = logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    db_path = config.tracking.db
    if not os.path.isabs(db_path):
        db_path = os.path.join(_HERE, db_path)
    tracker = RequestTracker(db_path) if config.tracking.enabled else None
    app = create_app(config, tracker)

    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
