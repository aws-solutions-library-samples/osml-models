#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import json
import os
import uuid
import warnings
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures.instances import Instances
from flask import Request, Response, request
from osgeo import gdal

from aws.osml.models import build_flask_app, build_logger, setup_server

ENABLE_SEGMENTATION = os.environ.get("ENABLE_SEGMENTATION", "False").lower() == "true"
ENABLE_FAULT_DETECTION = os.environ.get("ENABLE_FAULT_DETECTION", "False").lower() == "true"

# Enable exceptions for GDAL
gdal.UseExceptions()

# Create logger instance
logger = build_logger()

# Create our default flask app
app = build_flask_app(logger)


def build_predictor() -> DefaultPredictor:
    """
    Create a single detection predictor to detect aircraft
    :return: DefaultPredictor
    """
    # Load the prebuilt plane model w/ Detectron2
    cfg = get_cfg()
    # If we can't find a gpu
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
        app.logger.warning("GPU not found, running in CPU mode!")
    # Set to only expect one class (aircraft)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # Set the detection threshold to 90%
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    # Add project-specific config used for training to remove warnings
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # Path to the model weights
    cfg.MODEL.WEIGHTS = os.getenv(
        os.path.join("MODEL_WEIGHTS"), os.path.join("/home/osml-models/assets/", "model_weights.pth")
    )

    # Build the detectron2 default predictor
    return DefaultPredictor(cfg)


def mask_to_polygon(mask: torch.Tensor) -> List[List[float]]:
    """
    Convert a detectron2 instance mask in the form of a Tensor to a list-form polygon representing the mask.

    :param mask: A detectron2 instance mask in the form of a Tensor
    :return: A list form polygon representing the mask
    """
    # Convert tensor to numpy
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # Convert mask to 255 (fully white) and 0 (black)
    _, binary_mask = cv2.threshold(mask_np, 0.5, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Simplify contour if you want to save some cost in exchange
    # for reduced resolution on the masks
    # epsilon = 0.005 * cv2.arcLength(contours[0], True)
    # approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Convert contour to a list of lists
    polygon = [point[0].tolist() for point in contours[0]]

    return polygon


def instances_to_feature_collection(
    instances: Instances, image_id: Optional[str] = str(uuid.uuid4())
) -> Dict[str, Union[str, list]]:
    """
    Convert the gRPC response from the GetDetection call into a GeoJSON output.
    Each detection is a feature in the collection, including image coordinates,
    score, and type identifier as feature properties.

    :param instances: Detectron2 result instances
    :param image_id: Identifier for the processed image (optional)
    :return: FeatureCollection object containing detections
    """
    geojson_feature_collection_dict = {"type": "FeatureCollection", "features": []}
    if instances:
        # Get the bounding boxes for this image
        bboxes = instances.pred_boxes.tensor.cpu().numpy().tolist()

        # Get the scores for this image
        scores = instances.scores.cpu().numpy().tolist()

        # Default masks to None
        masks = None

        # Get the polygon masks for this image if segmentation is enabled
        if ENABLE_SEGMENTATION:
            masks = instances.pred_masks.cpu()

        for i in range(0, len(bboxes)):
            feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                "id": str(uuid.uuid4()),
                "properties": {
                    "bounds_imcoords": bboxes[i],
                    "detection_score": float(scores[i]),
                    "feature_types": {"aircraft": float(scores[i])},
                    "image_id": image_id,
                },
            }
            if masks is not None:
                feature["properties"]["geom_imcoords"] = mask_to_polygon(masks[i])
            app.logger.debug(feature)
            geojson_feature_collection_dict["features"].append(feature)
    else:
        app.logger.debug("No features found!")

    return geojson_feature_collection_dict


def request_to_instances(req: Request) -> Union[Instances, None]:
    """
    Use GDAL to open the image. The binary payload from the HTTP request is used to
    create an in-memory VFS for GDAL which is then opened to decode the image into
    a dataset which will give us access to a NumPy array for the pixels. Then
    use that image to create detectron2 detection instances.

    :param req: Request: the flask request object passed into the SM endpoint
    :return: Either a set of detectron2 detection instances or nothing
    """
    # Set up default variables
    temp_ds_name = "/vsimem/" + str(uuid.uuid4())
    gdal_dataset = None
    instances = None
    try:
        # Load the binary memory buffer sent to the model
        gdal.FileFromMemBuffer(temp_ds_name, req.get_data())
        gdal_dataset = gdal.Open(temp_ds_name)

        # Read GDAL dataset and convert to a numpy array
        image_array = gdal_dataset.ReadAsArray()

        # Check if all pixels are zero and raise an exception if so
        if ENABLE_FAULT_DETECTION:
            app.logger.debug(f"Image array min: {image_array.min()}, max: {image_array.max()}")
            if np.all(np.isclose(image_array, 0)):
                err = "All pixels in the image tile are set to 0."
                app.logger.error(err)
                raise Exception(err)

        # Handling of different image shapes
        if image_array.ndim == 2:  # For grayscale images without a channel dimension
            # Reshape to add a channel dimension and replicate across 3 channels for RGB
            image_array = np.stack([image_array] * 3, axis=0)
        elif image_array.shape[0] == 1:  # For grayscale images with a channel dimension
            # Replicate the single channel across 3 channels for RGB
            image_array = np.repeat(image_array, 3, axis=0)
        elif image_array.shape[0] == 4:  # For images with an alpha channel
            # Remove the alpha channel
            image_array = image_array[:3, :, :]

        # Conversion to uint8 (ensure this is done after ensuring 3 channels)
        image_array = (image_array * 255).astype(np.uint8)

        # Transpose the array from (channels, height, width) to (height, width, channels)
        image = np.transpose(image_array, (1, 2, 0))
        app.logger.debug(f"Running D2 on image array: {image}")

        # PyTorch can often give warnings about upcoming changes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            instances = aircraft_predictor(image)["instances"]
    except Exception as err:
        app.logger.error(f"Unable to load tile from request: {err}")
        raise err
    finally:
        try:
            if gdal_dataset is not None:
                if temp_ds_name is not None:
                    gdal.Unlink(temp_ds_name)
                del gdal_dataset
        except Exception as err:
            app.logger.warning(f"Unable to cleanup gdal dataset: {err}")

    return instances


# Build our aircraft predictor
aircraft_predictor = build_predictor()


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
        # Load the image into memory and get detection instances
        app.logger.debug("Loading image request.")
        instances = request_to_instances(request)

        # Generate a geojson feature collection that we can return
        geojson_detects = instances_to_feature_collection(instances)
        app.logger.debug(f"Sending geojson to requester: {json.dumps(geojson_detects)}")

        # Send back the detections
        return Response(response=json.dumps(geojson_detects), status=200)
    except Exception as err:
        app.logger.debug(err)
        return Response(response="Unable to process request!", status=500)


# pragma: no cover
if __name__ == "__main__":
    setup_server(app)
