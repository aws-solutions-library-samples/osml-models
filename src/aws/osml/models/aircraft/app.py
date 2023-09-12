#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import os
from json import dumps
from typing import List

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from flask import Flask, Response, request
from osgeo import gdal

from aws.osml.models import detect_to_geojson, load_image, setup_server, mask_to_polygon

ENABLE_SEGMENTATION = bool(os.environ.get("ENABLE_SEGMENTATION", False))

app = Flask(__name__)


def build_predictor() -> DefaultPredictor:
    """
    Create a single detection predictor to detect aircraft
    :return: DefaultPredictor
    """
    # load the prebuilt plane model w/ Detectron2
    cfg = get_cfg()
    # if we can't find a gpu
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    # set the number of classes we expect
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # add project-specific config used for training to remove warnings
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # path to the model weights we trained
    cfg.MODEL.WEIGHTS = os.getenv(os.path.join("MODEL_WEIGHTS"), os.path.join("/home/assets/", "model_weights.pth"))

    # build the default predictor
    return DefaultPredictor(cfg)


# build the default predictor
plane_predictor = build_predictor()


@app.route("/ping", methods=["GET"])
def healthcheck() -> Response:
    """
    This is a health check that will always pass since this is a stub model.

    :return: Successful status code (200) indicates all is well
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
    app.logger.debug("Invoking model endpoint using the Detectron2 Aircraft Model!")
    try:
        # Load the image to get its dimensions
        app.logger.debug("Loading image request.")
        ds = load_image(request)

        # If it failed to load return the failed Response
        if ds is None:
            return Response(response="Unable to parse image from request!", status=400)

        # Pull out the width and height from the dataset
        width, height = ds.RasterXSize, ds.RasterYSize

        app.logger.debug(f"Processing image of size: {width}x{height} with flood model.")

        # Set up a FeatureCollection to store our generated Features
        json_results = {"type": "FeatureCollection", "features": []}

        # set up a place holder array for our detections
        detects: List[dict] = []

        # path to tmp file
        tmp_file = "tmp.tif"

        # convert the GDAL dataset into a temporary file
        gdal.Translate(tmp_file, ds)

        if os.path.isfile(tmp_file):
            # load it into cv2
            im = cv2.imread(tmp_file)

            # grab detections
            instances = plane_predictor(im)["instances"]

            # if we had detection instances then add them
            if instances is not None:
                app.logger.debug(f"Found {len(instances)} detections in image.")

                # get the bboxes for this image
                boxes = instances.pred_boxes.tensor.cpu().numpy().tolist()

                # get the scores for this image
                scores = instances.scores.cpu().numpy().tolist()

                if ENABLE_SEGMENTATION is True:
                    # get the polygons for this image
                    masks = instances.pred_masks.cpu().numpy()
                else:
                    masks = None
                # concert our bboxes to geojson FeatureCollections
                for i in range(0, len(boxes)):
                    if masks is not None:
                        detects.append(detect_to_geojson(boxes[i], mask_to_polygon(masks[i]), scores[i], "airplane"))
                    else:
                        detects.append(detect_to_geojson(boxes[i], None, scores[i], "airplane"))

        # generate a plane detection from a D2 pretrained model
        json_results["features"].extend(detects)

        app.logger.debug("Sending success response to requester.")
        # send back the detections
        return Response(response=dumps(json_results), status=200)

    except Exception as err:
        app.logger.debug(err)
        return Response(response="Unable to process request!", status=500)
    finally:
        del ds


# pragma: no cover
if __name__ == "__main__":
    setup_server(app)
