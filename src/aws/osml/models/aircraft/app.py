#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import os
from json import dumps
from secrets import token_hex
from typing import List

import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from flask import Flask, Response, request
from osgeo import gdal

from aws.osml.models import detect_to_geojson, load_image, mask_to_polygon, setup_server

ENABLE_SEGMENTATION = bool(os.environ.get("ENABLE_SEGMENTATION", False))

# enable exceptions for GDAL
gdal.UseExceptions()

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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
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
    # path to tmp file
    tmp_file = f"tmp-{token_hex(16)}.tif"
    temp_ds_name = None
    try:
        # Load the image to get its dimensions
        app.logger.debug("Loading image request.")
        ds, temp_ds_name = load_image(request)

        # If it failed to load return the failed Response
        if ds is None:
            gdal.Unlink(temp_ds_name)
            return Response(response="Unable to parse image from request!", status=400)

        # Pull out the width and height from the dataset
        width, height = ds.RasterXSize, ds.RasterYSize

        app.logger.debug(f"Processing image of size: {width}x{height} with flood model.")

        # Set up a FeatureCollection to store our generated Features
        json_results = {"type": "FeatureCollection", "features": []}

        # set up a place holder array for our detections
        detects: List[dict] = []

        try:
            # convert the GDAL dataset into a temporary file
            gdal.Translate(tmp_file, ds)
        except Exception as err:
            app.logger.debug(err)
            return Response(response=f"Unable to write file from GDAL ds! Error: {err}", status=400)
        finally:
            # If it failed to load return the failed Response
            if not os.path.isfile(tmp_file):
                return Response(response="Unable to write file from GDAL ds!", status=400)

        # load it into cv2
        im = cv2.imread(tmp_file, cv2.IMREAD_LOAD_GDAL)

        # grab detections
        instances = plane_predictor(im)["instances"]

        # if we had detection instances then add them
        if instances is not None:
            app.logger.debug(f"Found {len(instances)} detections in image.")

            # get the bounding boxes for this image
            boxes = instances.pred_boxes.tensor.cpu().numpy().tolist()

            # get the scores for this image
            scores = instances.scores.cpu().numpy().tolist()

            if ENABLE_SEGMENTATION is True:
                # get the polygons for this image
                masks = instances.pred_masks.cpu().numpy()
            else:
                masks = None
            # concert our bounding boxes to geojson FeatureCollections
            for i in range(0, len(boxes)):
                if masks is not None:
                    app.logger.debug(f"Found {len(masks)} masks in image.")
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
        # clean up the dataset
        del ds
        if os.path.isfile(tmp_file):
            os.remove(tmp_file)
        if temp_ds_name is not None:
            gdal.Unlink(temp_ds_name)


# pragma: no cover
if __name__ == "__main__":
    setup_server(app)
