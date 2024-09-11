#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import json
import os
import unittest

from moto import mock_aws


@mock_aws
class AircraftModelTest(unittest.TestCase):
    """
    Unit test case for testing Flask endpoints in the aircraft detection app.

    This test suite uses the unittest framework and mocks AWS services using `moto`.
    Environment variables are set for the segmentation feature. Each test case
    simulates HTTP requests and verifies responses from the app.
    """

    # Enable segmentation for testing
    os.environ["ENABLE_SEGMENTATION"] = "True"

    def setUp(self):
        """
        Set up test environment before each test case.

        This method patches the Docker container ID for logging purposes,
        initializes the application context, and creates a test client.
        """
        # Initialize Flask application context and test client
        from aws.osml.models.aircraft.app import app

        self.ctx = app.app_context()
        self.ctx.push()
        self.client = app.test_client()

    def tearDown(self):
        """
        Clean up the test environment after each test case.

        This method pops the Flask application context after the test case is executed.
        """
        self.ctx.pop()

    def test_ping(self):
        """
        Test the `/ping` endpoint to check if the application is running.

        Sends a GET request to `/ping` and verifies that the response status code is 200.
        """
        response = self.client.get("/ping")
        assert response.status_code == 200

    @staticmethod
    def compare_two_geojson_results(actual_geojson_result, expected_json_result):
        """
        Helper method to compare two GeoJSON results.

        :param actual_geojson_result: GeoJSON result returned from the prediction model.
        :param expected_json_result: Expected GeoJSON result for comparison.

        The method checks the `type` and `features` fields, and compares the geometries of the features.
        """
        assert actual_geojson_result.get("type") == expected_json_result.get("type")
        assert len(actual_geojson_result.get("features")) == len(expected_json_result.get("features"))

        for actual_result, expected_result in zip(
            actual_geojson_result.get("features"), expected_json_result.get("features")
        ):
            assert actual_result.get("geometry") == expected_result.get("geometry")

    def test_predict_aircraft_model(self):
        """
        Test the aircraft detection model's prediction using a sample image.

        This test sends a sample image in a POST request to the `/invocations` endpoint
        and verifies that the GeoJSON result matches the expected model output.

        It uses `compare_two_geojson_results` to assert that the predicted result is correct.
        """
        data_binary = open("assets/images/2_planes.tiff", "rb")
        response = self.client.post("/invocations", data=data_binary)

        assert response.status_code == 200
        actual_geojson_result = json.loads(response.data)

        # Load expected GeoJSON result for comparison
        with open("test/sample_data/sample_aircraft_model_output.geojson", "r") as model_output_geojson:
            expected_json_result = json.loads(model_output_geojson.read())
            self.compare_two_geojson_results(actual_geojson_result, expected_json_result)

    def test_predict_bad_data_file(self):
        """
        Test the model's response to invalid data input.

        Sends a `None` object in the POST request to the `/invocations` endpoint
        and verifies that the response status code is 400 (Bad Request).
        """
        data_binary = None
        response = self.client.post("/invocations", data=data_binary)

        assert response.status_code == 400
