import atexit
import signal
import subprocess
import sys


def test_initialize_worker_uses_importable_module_name_independent_of_cwd(
    monkeypatch, tmp_path
):
    from style_bert_vits2.nlp.japanese import pyopenjtalk_worker

    attempts = 0
    popen_args: list[str] = []

    def create_worker_client(port: int):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError
        return object()

    def capture_popen(args: list[str], **kwargs):
        popen_args.extend(args)
        return object()

    monkeypatch.setattr(pyopenjtalk_worker, "WORKER_CLIENT", None)
    monkeypatch.setattr(pyopenjtalk_worker, "WorkerClient", create_worker_client)
    monkeypatch.setattr(subprocess, "Popen", capture_popen)
    monkeypatch.setattr(atexit, "register", lambda function: None)
    monkeypatch.setattr(signal, "signal", lambda signum, handler: None)
    monkeypatch.chdir(tmp_path)

    pyopenjtalk_worker.initialize_worker(port=12345)

    assert popen_args == [
        sys.executable,
        "-m",
        "style_bert_vits2.nlp.japanese.pyopenjtalk_worker",
        "--port",
        "12345",
    ]
