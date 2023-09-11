#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import argparse
import logging
from secrets import token_hex
from typing import List, Optional, Union

from flask import Flask, Request
from osgeo import gdal

# for segmentation 
import numpy as np
import cv2

def setup_server(app: Flask):
    """
    The assumption is that this script will be the ENTRYPOINT for the inference
    container. SageMaker will launch the container with the "serve" argument. We
    also have the option of using multiple models from this single container,
    Only one model will be active at a time (i.e. this is not a Multi Model Server)
    so it can be selected by name using the "model" parameter.

    :param app: The flask application to set up
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="serve")
    parser.add_argument("-v", "--verbose", default=False)
    args = parser.parse_args()

    # Set up the logging for the Flask application
    # and log the startup information
    configure_logging(args.verbose)
    logging.info("Initializing REST Model Server...")
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    # Start the simple web application server using Waitress.
    # Flask's app.run() is only intended to be used in development
    # mode so this provides a solution for hosting the application.
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080, clear_untrusted_proxy_headers=True)


def configure_logging(verbose: bool = False) -> None:
    """
    Configure application logging. Note the timestamp for the
    log record is available in CloudWatch, if this is used in a
    non-AWS environment you can add %(asctime)s to the start of
    the format string.

    :param verbose: True if the DEBUG log level should be used; defaults to False
    :return: None
    """
    logging_level = logging.INFO
    if verbose:
        logging_level = logging.DEBUG

    logging.basicConfig(level=logging_level, format="%(levelname)-8s : %(message)s")


def load_image(request: Request) -> Union[gdal.Dataset, None]:
    """
    Use GDAL to open the image. The binary payload from the HTTP request is used to
    create an in-memory VFS for GDAL which is then opened to decode the image into
    a dataset which will give us access to a NumPy array for the pixels.

    :param request: Request: the flask request object passed into the SM endpoint
    :return: Union[gdal.Dataset, None]: either the gdal dataset or nothing
    """
    try:
        temp_ds_name = "/vsimem/" + token_hex(16)
        gdal.FileFromMemBuffer(temp_ds_name, request.get_data())
        ds = gdal.Open(temp_ds_name)
        if not ds:
            logging.debug("Unable to parse request payload using GDAL.")
            return None
        return ds
    except Exception as err:
        logging.debug("There was an exception when trying to load image!")
        logging.debug(err)
        return None


def detect_to_geojson_dict(
    fixed_object_bbox: List[float], detection_score: Optional[float] = 1.0, detection_type: Optional[str] = "sample_object"
) -> dict:
    """
    Convert the bbox object into a sample GeoJSON formatted detection. Note
    that the world coordinates are not normally provided by the model container,
    so they're defaulted to 0,0 here since GeoJSON features require a geometry.

    :param detection_type: the class of the detection
    :param detection_score: the confidence score of the detection
    :param fixed_object_bbox:  Bounding box to transform into a geojson feature
    :return: dict: dictionary representation of a geojson feature
    """
    return {
        "geometry": {"coordinates": [0.0, 0.0], "type": "Point"},
        "id": token_hex(16),
        "properties": {
            "bounds_imcoords": fixed_object_bbox,
            "detection_score": detection_score,
            "feature_types": {detection_type: detection_score},
            "image_id": token_hex(16),
        },
        "type": "Feature",
    }


def convert_mask_to_polygon(mask: np.ndarray) -> List[float]:
    """
    converts a detectron 2 mask (or really any image) into a polygonal line.

    note this version only returns continuous polygons per detection with no holes.
    handling other topologies is left for the future, note 'has_holes' as a hook & logged warning
    :param mask: the instance mask from forward pass
    """

    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    # per https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
    # NONE is default, but we don't need every pixel. after testing, using TC89_KCOS as it's 1/5 of coords 
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS) 
    hierarchy = res[-1] 
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    if has_holes:
        logging.warning(f" detection mask has holes that we ignore")
    res = res[-2]
    res = [x.flatten() for x in res]

    # ok now let's convert those to 'coordinates'
    length = int(res[0].shape[0]/2)
    coords = res[0].reshape((length, 2))

    # and lets close the polygon to ensure geojson standard
    poly = coords.tolist()
    poly.append(poly[0])
    
    return poly


# this method is used in the 'airplane-segmentation' model.
def detect_and_mask_to_geojson_dict(
    fixed_object_bbox: List[float], fixed_object_mask: np.ndarray,
    detection_score: Optional[float] = 1.0, detection_type: Optional[str] = "sample_object"
) -> dict:
    """
    Convert the bbox object into a sample GeoJSON formatted detection. Note
    that the world coordinates are not normally provided by the model container,
    so they're defaulted to 0,0 here since GeoJSON features require a geometry.

    :param detection_type: the class of the detection
    :param detection_score: the confidence score of the detection
    :param fixed_object_bbox:  Bounding box to transform into a geojson feature
    :param fixed_object_mask:  Detection Mask to transform into a polygonal geojson feature
    :return: dict: dictionary representation of a geojson feature
    """

    return {
        "geometry": {"coordinates": [0.0, 0.0], "type": "Point"},
        "id": token_hex(16),
        "properties": {
            "bounds_imcoords": fixed_object_bbox,
            "polygon_imcoords": convert_mask_to_polygon(fixed_object_mask),
            "detection_score": detection_score,
            "feature_types": {detection_type: detection_score},
            "image_id": token_hex(16),
        },
        "type": "Feature",
    }

# this method is used in the 'centerpoint-segmentation' mock model.
# only difference is this accepts a polygonal list vs a mask as above

def detect_to_geojson_segmentation_dict(
    fixed_object_bbox: List[float], fixed_object_polygon: List[float],
    detection_score: Optional[float] = 1.0, detection_type: Optional[str] = "sample_object"
) -> dict:
    """
    Convert the bbox object into a sample GeoJSON formatted detection. Note
    that the world coordinates are not normally provided by the model container,
    so they're defaulted to 0,0 here since GeoJSON features require a geometry.

    :param detection_type: the class of the detection
    :param detection_score: the confidence score of the detection
    :param fixed_object_bbox:  Bounding box to transform into a geojson feature
    :param fixed_object_polygon:  Polygonal object to transform into a geojson feature
    :return: dict: dictionary representation of a geojson feature
    """
    return {
        "geometry": {"coordinates": [0.0, 0.0], "type": "Point"},
        "id": token_hex(16),
        "properties": {
            "bounds_imcoords": fixed_object_bbox,
            "polygon_imcoords": fixed_object_polygon,
            "detection_score": detection_score,
            "feature_types": {detection_type: detection_score},
            "image_id": token_hex(16),
        },
        "type": "Feature",
    }