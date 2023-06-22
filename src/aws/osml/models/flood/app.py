#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import math
import os
from json import dumps
from random import randrange
from typing import List

from flask import Flask, Response, request

from aws.osml.models.server_utils import detect_to_geojson_dict, load_image, setup_server

app = Flask(__name__)

# Optional ENV configurations
BBOX_PERCENTAGE = float(os.environ.get("BBOX_PERCENTAGE", 0.1))
FLOOD_VOLUME = int(os.environ.get("FLOOD_VOLUME", 100))


def gen_flood_detects(flood_volume: int, height: int, width: int, bbox_percentage: float) -> List[dict]:
    """
    Generate a random detection within the input image given a buffer percentage that
    limits the bounding boxes we generate to always fall within the image bounds.

    :param bbox_percentage: The size of the bounding box to produce.
    :param width: Width of the image tile.
    :param height: Height of the image tile.
    :param flood_volume: Number of random detections to generate
    :return: Union[gdal.Dataset, None]: either the gdal dataset or nothing
    """
    detects: List[dict] = []
    for i in range(flood_volume):
        fixed_object_size_xy = math.ceil(width * bbox_percentage), math.ceil(height * bbox_percentage)
        gen_x = randrange(fixed_object_size_xy[0], width - fixed_object_size_xy[0])
        gen_y = randrange(fixed_object_size_xy[1], height - fixed_object_size_xy[1])
        fixed_object_bbox = [
            gen_x - fixed_object_size_xy[0],
            gen_y - fixed_object_size_xy[1],
            gen_x + fixed_object_size_xy[0],
            gen_y + fixed_object_size_xy[1],
        ]
        detects.append(detect_to_geojson_dict(fixed_object_bbox))

    return detects


@app.route("/ping", methods=["GET"])
def healthcheck() -> Response:
    """
    This is a health check that will always pass since this is a stub model.

    :return: a successful status code (200) indicates all is well
    """
    app.logger.debug("Responding to health check")
    return Response(response="\n", status=200)


@app.route("/invocations", methods=["POST"])
def predict() -> Response:
    """
    This is the model invocation endpoint for the model container's REST
    API. The binary payload, in this case an image, is taken from the request
    parsed to ensure it is a valid image. This is a stub implementation that
    will always return fixed set of detections for a valid input image.

    :return: Response: Contains the GeoJSON results or an error status
    """
    app.logger.debug("Invoking flood model endpoint!")

    try:
        # load the image to get its dimensions
        ds = load_image(request)

        # if it failed to load return the failed Response
        if ds is None:
            return Response(response="Unable to parse image from request!", status=400)

        # pull out the width and height from the dataset
        width, height = ds.RasterXSize, ds.RasterYSize

        # set up a FeatureCollection to store our generated Features
        logging.debug(f"Processing image of size: {width}x{height} with flood model.")
        json_results = {"type": "FeatureCollection", "features": []}

        # generate random flood detections
        json_results["features"].extend(gen_flood_detects(FLOOD_VOLUME, width, height, BBOX_PERCENTAGE))

        # send back the detections
        return Response(response=dumps(json_results), status=200)

    except Exception as err:
        app.logger.warning("Image could not be processed by the test model server.", exc_info=True)
        app.logger.warning(err)
        return Response(response="Unable to process request.", status=500)

    finally:
        del ds  # Cleans up the dataset


if __name__ == "__main__":  # pragma: no cover
    setup_server(app)
