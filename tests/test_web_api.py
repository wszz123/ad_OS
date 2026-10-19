import json
import threading
import time
import unittest
from urllib.request import Request, urlopen

from coop.web_server import Handler, ThreadingHTTPServer, start_engine


class WebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        start_engine("fixture")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        with urlopen(self.base + path, timeout=10) as response:
            return response.status, json.loads(response.read())

    def post(self, path, payload):
        request = Request(self.base + path, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())

    def test_health_presets_and_fixture_run(self):
        status, health = self.get("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["mode"], "fixture")
        _, presets = self.get("/api/presets")
        self.assertGreaterEqual(len(presets["presets"]), 10)
        status, run = self.post("/api/run", {"task_id": "orion-01", "mode": "protocol", "use_vector": True})
        self.assertEqual(status, 200)
        self.assertEqual(run["status"], "complete")
        self.assertEqual(run["result"]["state_encoding"], "float16")
        self.assertEqual(run["result"]["roles"][-1]["status"], "done")

    def test_manual_request_rejects_invalid_entity_and_unknown_run(self):
        request = Request(self.base + "/api/run", data=json.dumps({
            "query": "x", "entity": "Unknown", "inputs": {}, "mode": "protocol"
        }).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(Exception):
            urlopen(request, timeout=10)
        request = Request(self.base + "/api/runs/missing")
        with self.assertRaises(Exception):
            urlopen(request, timeout=10)


if __name__ == "__main__":
    unittest.main()
