from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from engine.models import BoundingBox, ImageRecord, ImageStatus, PersonDetection, RankingResult
from engine.project_export import export_project


class ProjectExportTests(unittest.TestCase):
    def test_export_is_an_image_mosaic_viewer_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "person.jpg"
            Image.new("RGB", (1000, 800), "white").save(source, quality=90)

            record = ImageRecord(
                path=str(source),
                filename=source.name,
                width=1000,
                height=800,
                status=ImageStatus.SELECTED,
                detections=[
                    PersonDetection(
                        bbox=BoundingBox(x=200, y=100, width=400, height=500),
                        confidence=0.95,
                        relative_size=0.25,
                        is_main=True,
                    )
                ],
                ranking=RankingResult(final_score=92.0, rank=1),
            )

            output = export_project(str(root / "out"), [record], (1920, 1080), 40)
            data = json.loads(output.read_text(encoding="utf-8"))

            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            entry = data[0]
            self.assertEqual(
                set(entry),
                {"type", "filename", "coords", "zoom", "canvas_size"},
            )
            self.assertEqual(entry["type"], "body")
            self.assertEqual(entry["filename"], "selected/001_person.jpg")
            self.assertEqual(entry["canvas_size"], [1920, 1080])
            self.assertEqual(entry["zoom"], 0.5)
            self.assertEqual(len(entry["coords"]), 4)
            self.assertTrue(all(0.0 <= float(v) <= 1.0 for v in entry["coords"]))

            self.assertTrue((output.parent / entry["filename"]).is_file())
            self.assertTrue((output.parent / "ai_mosaic_metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
