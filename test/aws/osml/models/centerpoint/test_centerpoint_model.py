#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import json
import os
import unittest


class AppTestCase(unittest.TestCase):
    os.environ["BBOX_PERCENTAGE"] = "0.1"

    def setUp(self):
        from aws.osml.models.centerpoint import app

        self.ctx = app.app_context()
        self.ctx.push()
        self.client = app.test_client()

    def tearDown(self):
        self.ctx.pop()

    def test_ping(self):
        response = self.client.get("/ping")
        assert response.status_code == 200

    def compare_two_geojson_results(self, actual_geojson_result, expected_json_result):
        assert actual_geojson_result.get("type") == expected_json_result.get("type")
        assert len(actual_geojson_result.get("features")) == len(expected_json_result.get("features"))

        for actual_result, expected_result in zip(
            actual_geojson_result.get("features"), expected_json_result.get("features")
        ):
            assert actual_result.get("geometry") == expected_result.get("geometry")

            # Current issue is that comparing both geojson files will fail due to unique image_id
            # To overcome that issue, overwrite expected image_id with actual image_id
            actual_image_id = actual_result["properties"]["image_id"]
            expected_result["properties"]["image_id"] = actual_image_id

            assert actual_result.get("properties") == expected_result.get("properties")

    def test_predict_center_point_model(self):
        data_binary = open("assets/images/2_planes.tiff", "rb")
        response = self.client.post("/invocations", data=data_binary)

        assert response.status_code == 200

        sample_output = "test/sample_data/sample_center_point_model_output.geojson"
        with open(sample_output, "r") as model_output_geojson:
            expected_json_result = json.loads(model_output_geojson.read())

        actual_geojson_result = json.loads(response.data)
        self.compare_two_geojson_results(actual_geojson_result, expected_json_result)

    def test_predict_bad_data_file(self):
        data_binary = None
        response = self.client.post("/invocations", data=data_binary)

        assert response.status_code == 400
