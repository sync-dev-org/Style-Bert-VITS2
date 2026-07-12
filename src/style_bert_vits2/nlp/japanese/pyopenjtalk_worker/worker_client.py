import socket
import threading
from typing import Any, cast

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_common import (
    RequestType,
    receive_data,
    send_data,
)


class WorkerClient:
    """pyopenjtalk worker client"""

    def __init__(self, port: int) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # timeout: 60 seconds
        sock.settimeout(60)
        sock.connect((socket.gethostname(), port))
        self.sock = sock
        self._socket_lock = threading.Lock()

    def __enter__(self) -> "WorkerClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self.sock.close()

    def _request(self, data: dict[str, Any]) -> dict[str, Any]:
        with self._socket_lock:
            logger.trace(f"client sends request: {data}")
            send_data(self.sock, data)
            logger.trace("client sent request successfully")
            response = receive_data(self.sock)
            logger.trace(f"client received response: {response}")
            return response

    def dispatch_pyopenjtalk(self, func: str, *args: Any, **kwargs: Any) -> Any:
        data = {
            "request-type": RequestType.PYOPENJTALK,
            "func": func,
            "args": args,
            "kwargs": kwargs,
        }
        response = self._request(data)
        return response.get("return")

    def status(self) -> int:
        data = {"request-type": RequestType.STATUS}
        response = self._request(data)
        return cast(int, response.get("client-count"))

    def quit_server(self) -> None:
        data = {"request-type": RequestType.QUIT_SERVER}
        self._request(data)
