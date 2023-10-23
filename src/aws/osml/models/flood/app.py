#  Copyright 2023 Amazon.com, Inc. or its affiliates.
import json
import logging
import math
import os
from random import randrange
from secrets import token_hex
from typing import Dict, Union

from flask import Flask, Response, request
from osgeo import gdal

from aws.osml.models import detect_to_feature
from aws.osml.models.server_utils import setup_server

app = Flask(__name__)
app.logger.setLevel(logging.ERROR)

# Optional ENV configurations
BBOX_PERCENTAGE = float(os.environ.get("BBOX_PERCENTAGE", 0.1))
FLOOD_VOLUME = int(os.environ.get("FLOOD_VOLUME", 100))


def gen_flood_detects(height: int, width: int, bbox_percentage: float) -> Dict[str, Union[str, list]]:
    """
    Generate a random detection within the input image given a buffer percentage that
    limits the bounding boxes we generate to always fall within the image bounds.

    :param bbox_percentage: The size of the bounding box to produce.
    :param width: Width of the image tile.
    :param height: Height of the image tile.
    :return: Union[gdal.Dataset, None]: either the gdal dataset or nothing
    """
    geojson_features = []
    for i in range(FLOOD_VOLUME):
        fixed_object_size_xy = math.ceil(width * bbox_percentage), math.ceil(height * bbox_percentage)
        gen_x = randrange(fixed_object_size_xy[0], width - fixed_object_size_xy[0])
        gen_y = randrange(fixed_object_size_xy[1], height - fixed_object_size_xy[1])
        fixed_object_bbox = [
            gen_x - fixed_object_size_xy[0],
            gen_y - fixed_object_size_xy[1],
            gen_x + fixed_object_size_xy[0],
            gen_y + fixed_object_size_xy[1],
        ]
        fixed_object_mask = [
            [gen_x - fixed_object_size_xy[0], gen_y + fixed_object_size_xy[1]],
            [gen_y - fixed_object_size_xy[0], gen_x + fixed_object_size_xy[0]],
            [gen_x + fixed_object_size_xy[0], gen_y + fixed_object_size_xy[1]],
            [gen_y + fixed_object_size_xy[1], gen_x + fixed_object_size_xy[0]],
        ]
        feature = detect_to_feature(fixed_object_bbox, fixed_object_mask)
        geojson_features.append(feature)

    geojson_feature_collection_dict = {"type": "FeatureCollection", "features": geojson_features}

    return geojson_feature_collection_dict


@app.route("/ping", methods=["GET"])
def healthcheck() -> Response:
    """
    This is a health check that will always pass since this is a stub model.

    :return: A successful status code (200) indicates all is well
    """
    app.logger.debug("Responding to health check")
    return Response(response="\n", status=200)


@app.route("/invocations", methods=["POST"])
def predict() -> Response:
    """
    This is the model invocation endpoint for the model container's REST
    API. The binary payload, in this case an image, is taken from the request
    parsed to ensure it is a valid image. This is a stub implementation that
    will always return a fixed set of detections for a valid input image.

    :return: Response: Contains the GeoJSON results or an error status
    """
    app.logger.debug("Invoking flood model endpoint!")
    temp_ds_name = "/vsimem/" + token_hex(16)
    gdal_dataset = None
    try:
        # load the file from the request memory buffer
        gdal.FileFromMemBuffer(temp_ds_name, request.get_data())
        try:
            gdal_dataset = gdal.Open(temp_ds_name)

        # if it failed to load return the failed Response
        except RuntimeError:
            return Response(response="Unable to parse image from request!", status=400)

        # generate random flood detections
        geojson_detects = gen_flood_detects(gdal_dataset.RasterXSize, gdal_dataset.RasterYSize, BBOX_PERCENTAGE)

        # send back the detections
        return Response(response=json.dumps(geojson_detects), status=200)

    except Exception as err:
        app.logger.warning("Image could not be processed by the test model server.", exc_info=True)
        app.logger.warning(err)
        return Response(response="Unable to process request.", status=500)

    finally:
        if gdal_dataset is not None:
            if temp_ds_name is not None:
                gdal.Unlink(temp_ds_name)
            del gdal_dataset


if __name__ == "__main__":  # pragma: no cover
    setup_server(app)
