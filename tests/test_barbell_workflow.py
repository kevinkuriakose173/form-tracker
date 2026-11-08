import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from barbell_workflow import (build, check_sessions, file_hash, summarize_metrics,
                              validate_version, write_json, yolo_label)


class DatasetTests(unittest.TestCase):
    def row(self, identity="one", split="train", session="session_1", video="hash_1"):
        return {"id":identity,"split":split,"session":session,"video_sha256":video,
                "width":100,"height":200,"frame":0,"timestamp_s":0.,"image":f"raw/{identity}.jpg"}

    def test_video_cannot_be_reused_in_another_split_with_another_session_name(self):
        with self.assertRaisesRegex(ValueError,"leakage"):
            check_sessions([self.row(),self.row("two","test","other_session")])

    def test_different_clips_from_same_session_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError,"leakage"):
            check_sessions([self.row(),self.row("two","val",video="different_hash")])

    def test_normalized_label_contains_box_and_one_visible_point(self):
        annotation={"state":"visible","bbox":[20,40,60,120],"keypoint":[40,80]}
        fields=yolo_label(self.row(),annotation).split()
        self.assertEqual(len(fields),8)
        self.assertEqual(fields[0],"0")
        self.assertEqual(fields[-1],"2")
        np.testing.assert_allclose(list(map(float,fields[1:7])),[.4,.4,.4,.4,.4,.4])

    def test_invalid_and_nonfinite_labels_are_rejected(self):
        for point in ([90,190],[float("nan"),60]):
            with self.assertRaises(ValueError):
                yolo_label(self.row(),{"state":"visible","bbox":[20,40,60,120],"keypoint":point})

    def test_skip_and_negative_are_distinct(self):
        self.assertIsNone(yolo_label(self.row(),{"state":"skip"}))
        self.assertEqual(yolo_label(self.row(),{"state":"negative"}),"")

    def test_build_snapshot_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[self.row(),self.row("two","val","session_2","hash_2")]
            for row in rows:
                image=root/row["image"]
                image.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(image),np.zeros((200,100,3),np.uint8))
                write_json(root/"annotations"/(row["id"]+".json"),
                           {"state":"visible","bbox":[20,40,60,120],"keypoint":[40,80]})
            write_json(root/"manifest.json",rows)
            args=Namespace(dataset=root,name="v1")
            build(args)
            data=root/"versions/v1/data.yaml"
            version,prepared=validate_version(data)
            self.assertEqual(len(prepared),2)
            self.assertEqual(prepared[0]["annotation"]["keypoint"],[40,80])
            with self.assertRaisesRegex(ValueError,"already exists"):
                build(args)
            (version/"labels/train/one.txt").write_text("")
            with self.assertRaisesRegex(ValueError,"labels changed"):
                validate_version(data)


class MetricsTests(unittest.TestCase):
    def test_missing_detections_fail_tolerance_metric(self):
        records=[{"state":"visible","target":[0,0],"prediction":[3,4]},
                 {"state":"visible","target":[0,0],"prediction":None},
                 {"state":"visible","target":[0,0],"prediction":[6,8]},
                 {"state":"negative","target":None,"prediction":[5,5]},
                 {"state":"negative","target":None,"prediction":None}]
        metrics=summarize_metrics(records,5)
        self.assertAlmostEqual(metrics["detection_coverage"],2/3)
        self.assertAlmostEqual(metrics["fraction_visible_frames_within_tolerance"],1/3)
        self.assertEqual(metrics["median_localization_error_px_on_accepted"],7.5)
        self.assertEqual(metrics["false_positive_rate_on_negative_frames"],.5)

    def test_empty_predictions_do_not_become_zero_error(self):
        metrics=summarize_metrics([{"state":"visible","target":[5,5],"prediction":None}],5)
        self.assertIsNone(metrics["median_localization_error_px_on_accepted"])
        self.assertEqual(metrics["detection_coverage"],0)
        self.assertEqual(metrics["fraction_visible_frames_within_tolerance"],0)
        json.dumps(metrics,allow_nan=False)


if __name__=="__main__":
    unittest.main()
