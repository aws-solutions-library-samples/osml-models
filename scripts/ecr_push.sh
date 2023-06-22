#!/bin/bash

#
# Copyright 2023 Amazon.com, Inc. or its affiliates.
#

# This is a utility script to help building and uploading the OSML default test model to an accounts ECR
# repository deployed by the AWSOversightMLCDK package. While this script can be run without providing inputs,
# a user can pass in the following optional parameters to modify default behavior:
# $1 = group_name = log group to monitor
# $2 = time_window = window, in seconds, to filter log outputs
# $3 = region = region log group is contained in
#
# Example usage: ./ecr_push.sh sample-user us-east-1 sample-user-model-repo

# Grab user inputs or set default values
NAME="${1:-$USER}"
REGION="${2:-"us-west-2"}"
REPO=$NAME-"${3:-"sagemaker-model-repo"}"
TAG=$NAME-"${4:-"latest"}"

# Grab the account id for the loaded AWS credentials
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Login to to docker with garnered ECR credentials
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID".dkr.ecr."$REGION".amazonaws.com

# Build the container locally with docker
docker build -t "$REPO" .

# Tag the model for upload to ECR
docker tag "$REPO":latest "$ACCOUNT_ID".dkr.ecr."$REGION".amazonaws.com/"$REPO":latest

# Push to remote ECR repository
docker push "$ACCOUNT_ID".dkr.ecr."$REGION".amazonaws.com/"$REPO":"$TAG"
