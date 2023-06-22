# osml-models
This package contains sample models that can be used to test OversightML installations without incurring high compute
costs typically associated with complex Computer Vision models. These models implement an interface compatible with
SageMaker and are suitable for deployment as endpoints with CPU instances.

## Build and Local Testing

To build the container, it uses the default `Dockerfile` from the root of this repository. If you want to change to another `Dockerfile`, replace the `.` with the new `Dockerfile` path.
```bash
docker build . -t test-${MODEL_SELECTION}-model --target osml_model --build-arg MODEL_SELECTION=${MODEL_SELECTION}
```

**Note**: The `MODEL_SELECTION` environment variable can be used to pick the model to run. Currently, we support 3 different types of model and below are the appropriate naming convention:

- centerpoint
- flood
- aircraft

In one terminal, run the following command to start the server:
```bash
docker run -p 8080:8080 test-${MODEL_SELECTION}-model
```

In another terminal to invoke the rest server and return the inference on a single tile, run the following command from the root of this repository:

```bash
curl -I localhost:8080/ping
curl --request POST --data-binary "@<imagery file>" localhost:8080/invocations
```
 - Example: `curl --request POST --data-binary "@assets/images/2_planes.tiff" localhost:8080/invocations`

Executing above should return:

```
{"type": "FeatureCollection", "features": [{"geometry": {"coordinates": [0.0, 0.0], "type": "Point"}, "id": "7683a11e4c93f0332be9a4a53e0c6762", "properties": {"bounds_imcoords": [204.8, 204.8, 307.2, 307.2], "detection_score": 1.0, "feature_types": {"sample_object": 1.0}, "image_id": "8cdac8849cae2b4a8885c0dd0d34f722"}, "type": "Feature"}]}
```

## Build and Run Unit Tests

If you want to build and run the unit test:

```bash
export MODEL_SELECTION=<MODEL_SELECTION>
docker build . -t unit-test-${MODEL_SELECTION}-model --target unit_test --build-arg MODEL_SELECTION=${MODEL_SELECTION}
docker run --rm -it unit-test-${MODEL_SELECTION}-model:latest
```

It will show the code coverage for this repository.

To run the container in a test mode and work inside it (ensure you have built the test container):

```bash
docker run -u root -i -t $(docker images  | grep 'unit-test-<MODEL_SELECTION>-model' | awk '{print $3}') /bin/bash
```

Inside the docker container, you can run the pytest command:

```bash
python3 -m pytest -vv --cov-report=term-missing --cov=aws.osml.models.${MODEL_SELECTION} test/aws/osml/models/${MODEL_SELECTION}/
```

## Linting/Formatting

This package uses a number of tools to enforce formatting, linting, and general best practices:
- [Black](https://github.com/ambv/black) and [isort](https://github.com/timothycrosley/isort) for formatting with a max line length of 100
- [mypy](http://mypy-lang.org/) to enforce static type checking
- [flake8](https://pypi.python.org/pypi/flake8) to check pep8 compliance and logical errors in code
- [pre-commit](https://github.com/pre-commit/pre-commit-hooks) to install and control linters in githooks

```
pip install pre-commit
pre-commit install
```

Then run:

```
pre-commit run --all-files
```
