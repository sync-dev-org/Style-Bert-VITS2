import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from style_bert_vits2.nlp.japanese.pyopenjtalk_worker import worker_client
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_common import (
    receive_data,
    send_data,
)


def test_concurrent_dispatches_receive_their_own_responses(monkeypatch):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((socket.gethostname(), 0))
    server_socket.listen()
    server_errors: list[BaseException] = []

    def serve_requests() -> None:
        try:
            connection, _ = server_socket.accept()
            with connection:
                for _ in range(2):
                    request = receive_data(connection)
                    time.sleep(0.01)
                    send_data(connection, {"return": request["func"]})
        except BaseException as error:
            server_errors.append(error)

    server_thread = threading.Thread(target=serve_requests, daemon=True)
    server_thread.start()

    first_request_sent = threading.Event()
    second_request_sent = threading.Event()

    def schedule_send(sock: socket.socket, data: dict) -> None:
        send_data(sock, data)
        if data["func"] == "first":
            first_request_sent.set()
            second_request_sent.wait(timeout=0.2)
            time.sleep(0.05)
        else:
            second_request_sent.set()

    monkeypatch.setattr(worker_client, "send_data", schedule_send)

    try:
        with worker_client.WorkerClient(server_socket.getsockname()[1]) as client:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first_result = executor.submit(client.dispatch_pyopenjtalk, "first")
                assert first_request_sent.wait(timeout=1)
                second_result = executor.submit(client.dispatch_pyopenjtalk, "second")

                assert first_result.result() == "first"
                assert second_result.result() == "second"
    finally:
        server_socket.close()
        server_thread.join(timeout=1)

    assert not server_thread.is_alive()
    assert server_errors == []
