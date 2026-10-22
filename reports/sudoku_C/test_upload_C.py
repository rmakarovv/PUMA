"""Three checks of the import payload's public validation contract."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from upload_C import load_payload


class ImportValidationTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(__file__).with_name("runs.json")
        self.payload = json.loads(self.source.read_text())

    def validate_changed(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.json"
            path.write_text(json.dumps(payload))
            return load_payload(path)

    def test_complete_five_seed_history(self):
        runs = load_payload(self.source)["runs"]
        self.assertEqual({r["seed"] for r in runs}, {1000, 2000, 3000, 4000, 5000})
        self.assertTrue(all(r["history"][-1]["optimizer_step"] == 53570 for r in runs))

    def test_empty_history_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["runs"][0]["history"] = []
        with self.assertRaisesRegex(ValueError, "Empty or invalid history"):
            self.validate_changed(payload)

    def test_reordered_history_is_rejected(self):
        payload = copy.deepcopy(self.payload)
        history = payload["runs"][0]["history"]
        history[0], history[1] = history[1], history[0]
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            self.validate_changed(payload)


if __name__ == "__main__":
    unittest.main()
