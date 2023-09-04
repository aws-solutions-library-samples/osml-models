#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import os
from json import dumps

from flask import Flask, Response, request

from aws.osml.models.server_utils import detect_to_geojson_dict, load_image, setup_server
from aws.osml.models.server_utils import detect_to_geojson_segmentation_dict
from matplotlib.patches import Circle, CirclePolygon

app = Flask(__name__)

# Optional ENV configurations
BBOX_PERCENTAGE = float(os.environ.get("BBOX_PERCENTAGE", 0.1))
NUM_VERTICES = float(os.environ.get("NUM_VERTICES", 6))


def gen_center_point_and_polygon_detect(width: int, height: int, bbox_percentage: float) -> dict:
    """
    Create a single detection bbox and circular polygon that is at the center of and sized proportionally to the image

    :param bbox_percentage: the size of the bounding box and poly, relative to the image, to return
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

    """
    this draws a polygon with the same center
    and width and height percentage
    polygon can be a circle, or a hexagon, based on 
    """
    
    center = (width / 2, height / 2)
    radius = bbox_percentage
    number_of_vertices = 6 #20 is a nice circle, 3 is a triangle, etc
    circle = CirclePolygon(center,radius, resolution=number_of_vertices)
    poly_path = circle.get_path().vertices.tolist()
    # poly_path.append(coord[0]) # this may best be done downstream in real model, not necessary now thx to matplotlib.
    # TODO: discuss w/ Dr Duhe
    # that's in unit coordinates, let's project to the correct percentage of our image coordinates
    poly_scale = [bbox_percentage*width,bbox_percentage*height]
    scaled_circle = [(x*poly_scale[0],y*poly_scale[1]) for (x,y) in poly_path]

    return detect_to_geojson_segmentation_dict(fixed_object_bbox, scaled_circle)


@app.route("/ping", methods=["GET"])
def healthcheck() -> Response:
    """
    This is a health check that will always pass since this is a stub model.

    :return: a successful status code (200) indicates all is well
    """
    app.logger.info("Responding to health check from centerpoint segmentation")
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
    app.logger.debug("Invoking centerpoint model endpoint")

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

        # generate centerpoint detections with polygon
        json_results["features"].append(gen_center_point_and_polygon_detect(width, height, BBOX_PERCENTAGE))

        # Send back the detections
        return Response(response=dumps(json_results), status=200)

    except Exception as err:
        app.logger.warning("Image could not be processed by the centerpoint model server.", exc_info=True)
        app.logger.warning(err)
        return Response(response="Unable to process request.", status=500)

    finally:
        del ds  # Cleans up the dataset


if __name__ == "__main__":  # pragma: no cover
    setup_server(app)