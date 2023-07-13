#!/bin/bash

#
# Copyright 2023 Amazon.com, Inc. or its affiliates.
#

# This is a utility script to help building and uploading the OSML default test model to an accounts ECR
# repository deployed by the the cdk constructs or a custom ECR.
#
# Example usage: ./ecr_push.sh sample-user us-east-1 sample-user-model-repo

# Grab user inputs or set default values
NAME="${1:-$USER}"
REGION="${2:-"us-west-2"}"
REPO=$NAME-"${3:-"model-container"}"
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
