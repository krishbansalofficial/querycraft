import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from querycraft.demo import SQLITE_MAGIC, make_handler


@pytest.fixture
def server(tmp_path):
    calls = []

    def ask(path, question, hint):
        calls.append((path, question, hint))
        return {"state": "answer", "sql": "SELECT 1", "error": None, "columns": ["1"], "rows": [[1]],
                "votes": {"agree": 8, "n": 8, "needed": 6, "distinct": 1}}

    examples = {"toy": (str(tmp_path / "toy.sqlite"), [{"question": "q", "hint": ""}])}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ask, {"model": "m", "checkpoint": "c"}, examples, tmp_path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", calls
    srv.shutdown()


def _post(url, body, ctype="application/json"):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_info_ask_and_upload(server):
    url, calls = server
    with urllib.request.urlopen(url + "/") as r:
        assert b"<title>Querycraft</title>" in r.read()
    with urllib.request.urlopen(url + "/api/info") as r:
        assert json.loads(r.read())["databases"] == {"toy": [{"question": "q", "hint": ""}]}
    code, out = _post(url + "/api/ask", json.dumps({"db": "toy", "question": "how many?"}).encode())
    assert code == 200 and out["state"] == "answer" and calls[-1][1] == "how many?"
    assert _post(url + "/api/ask", json.dumps({"db": "nope", "question": "x"}).encode())[0] == 400
    assert _post(url + "/api/upload", b"not a database", "application/octet-stream")[0] == 400
    code, out = _post(url + "/api/upload", SQLITE_MAGIC + b"\0" * 100, "application/octet-stream")
    assert code == 200 and out["db"].startswith("upload:")
    _post(url + "/api/ask", json.dumps({"db": out["db"], "question": "y"}).encode())
    assert calls[-1][0].endswith(".sqlite")
