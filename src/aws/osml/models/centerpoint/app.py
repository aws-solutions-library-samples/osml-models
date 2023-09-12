#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import os
from json import dumps

from flask import Flask, Response, request
from matplotlib.patches import CirclePolygon

from aws.osml.models.server_utils import detect_to_geojson, load_image, setup_server

app = Flask(__name__)

# Optional ENV configurations
BBOX_PERCENTAGE = float(os.environ.get("BBOX_PERCENTAGE", 0.1))
ENABLE_SEGMENTATION = bool(os.environ.get("ENABLE_SEGMENTATION", False))


def gen_center_bbox(width: int, height: int, bbox_percentage: float) -> list:
    """
    Create a single detection bbox that is at the center of and sized proportionally to the image
    :param bbox_percentage: the size of the bounding box and poly, relative to the image, to return
    :param width: Raster width of the image passed in
    :param height: Raster height of the image passed in
    :return:
    """
    center_xy = width / 2, height / 2
    fixed_object_size_xy = width * bbox_percentage, height * bbox_percentage
    return [
        center_xy[0] - fixed_object_size_xy[0],
        center_xy[1] - fixed_object_size_xy[1],
        center_xy[0] + fixed_object_size_xy[0],
        center_xy[1] + fixed_object_size_xy[1],
    ]


def gen_center_polygon_detect(width: int, height: int, bbox_percentage: float) -> dict:
    """
    Create  circular polygon that is at the center of and sized proportionally to the bbox
    :param bbox_percentage: the size of the bounding box and poly, relative to the image, to return
    :param width: Raster width of the image passed in
    :param height: Raster height of the image passed in
    :return: geojson: Segmented polygon for center detection
    this draws a polygon with the same center
    and width and height percentage polygon can be a circle, or a hexagon, or triangle, etc. - based on the
    number_of_vertices there is a chance this is not centered as we'd like - meanwhile this will work as-is for initial
     OSML segmentation 'passthrough'

    """
    center = 0, 0
    center_xy = width / 2, height / 2
    fixed_object_bbox = gen_center_bbox(width, height, bbox_percentage)
    radius = bbox_percentage
    number_of_vertices = 6  # 20 is a nice circle, 3 is a triangle, etc
    circle = CirclePolygon(center, radius, resolution=number_of_vertices)
    poly_path = circle.get_path().vertices.tolist()
    poly_path.append(poly_path[0])  # this is part of CV model requirements, to have (only) closed polygons
    nonzero_circle = [((x + 1) / 2, (y + 1) / 2) for (x, y) in poly_path]  # this moves poly to nonzero 0-1 coords
    # let's project to the correct percentage of our image coordinates
    poly_scale = [bbox_percentage * width, bbox_percentage * height]
    # and do final scaling of coordinates, and w/h translation to ensure within bounds of the bbox
    scaled_circle = [(x * poly_scale[0] + center_xy[0], y * poly_scale[1] + center_xy[1]) for (x, y) in nonzero_circle]

    return detect_to_geojson(fixed_object_bbox, scaled_circle)


def gen_center_point_detect(width: int, height: int, bbox_percentage: float) -> dict:
    """
    Create a single detection bbox that is at the center of and sized proportionally to the image.

    :param bbox_percentage: Size of the bounding box, relative to the image, to return
    :param width: Raster width of the image passed in
    :param height: Raster height of the image passed in
    :return:
    """
    center_xy = width / 2, height / 2
    fixed_object_size_xy = width * bbox_percentage, height * bbox_percentage
    fixed_object_bbox = [
        center_xy[0] - fixed_object_size_xy[0],
        center_xy[1] - fixed_object_size_xy[1],
        center_xy[0] + fixed_object_size_xy[0],
        center_xy[1] + fixed_object_size_xy[1],
    ]

    return detect_to_geojson(fixed_object_bbox)


@app.route("/ping", methods=["GET"])
def healthcheck() -> Response:
    """
    This is a health check that will always pass since this is a stub model.

    :return: Response: Status code (200) indicates all is well
    """
    app.logger.debug("Responding to health check")
    return Response(response="\n", status=200)


@app.route("/invocations", methods=["POST"])
def predict() -> Response:
    """
    This is the model invocation endpoint for the model container's REST
    API. The binary payload, in this case an image, is taken from the request
    parsed to ensure it is a valid image. This is a stub implementation that
    will always return the fixed set of detections for a valid input image.

    :return: Response: Contains the GeoJSON results or an error status
    """
    app.logger.debug("Invoking centerpoint model endpoint")

    try:
        # load the image to get its dimensions
        ds = load_image(request)

        # sf it failed to load return the failed Response
        if ds is None:
            return Response(response="Unable to parse image from request!", status=400)

        # pull out the width and height from the dataset
        width, height = ds.RasterXSize, ds.RasterYSize

        # set up a FeatureCollection to store our generated Features
        logging.debug(f"Processing image of size: {width}x{height} with flood model.")
        json_results = {"type": "FeatureCollection", "features": []}

        if ENABLE_SEGMENTATION is True:
            json_results["features"].append(gen_center_polygon_detect(width, height, BBOX_PERCENTAGE))
        else:
            json_results["features"].append(gen_center_point_detect(width, height, BBOX_PERCENTAGE))

        # send back the detections
        return Response(response=dumps(json_results), status=200)

    except Exception as err:
        app.logger.warning("Image could not be processed by the centerpoint model server.", exc_info=True)
        app.logger.warning(err)
        return Response(response="Unable to process request.", status=500)

    finally:
        # cleans up the dataset
        del ds


# pragma: no cover
if __name__ == "__main__":
    setup_server(app)
