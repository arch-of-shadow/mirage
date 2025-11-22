#!/usr/bin/env bash
# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

echo "THIS SCRIPT MUST BE RUN FROM THE ROOT OF MIRAGE"
echo "==============================================="

IMAGE_NAME="$1"
CONTAINER_NAME="$2"
COMMAND="bash"
PWD=$(pwd)

WORKSPACE="/mirage"

echo "PWD: ${PWD}"
echo "WORKSPACE: ${WORKSPACE}"
echo "IMAGE NAME: ${IMAGE_NAME}"
echo "CONTAINER NAME: ${CONTAINER_NAME}"
echo "DOCKER BINARY: docker"



if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "Image '$IMAGE_NAME' does not exist locally."
  echo "Building image..."
  docker build -t ${IMAGE_NAME} ./docker/
else
  echo "Image '$IMAGE_NAME' found."
  echo "Skipping build."
fi


# Check if the container exists (running or stopped)
# -a: check all (including stopped)
# -q: quiet (print ID only)
# -f: filter by name (using regex ^...$ ensures exact match)
if [[ "$(docker ps -aq -f name="^/${CONTAINER_NAME}$")" ]]; then
    echo "Container '$CONTAINER_NAME' found."
    # Start the container (if it's already running, this does nothing but is safe)
    docker start "$CONTAINER_NAME"
    # Attach to the running container
    # Use Ctrl+P, Ctrl+Q to detach without stopping
    docker attach "$CONTAINER_NAME"
else
    echo "Container '$CONTAINER_NAME' not found."
    echo "Creating and running..."
    # Run a new container
    # -it: Interactive mode (crucial if you want to attach/type commands)
    docker run \
        -it --net=host \
        --name ${CONTAINER_NAME} \
        -v ${PWD}:/mirage \
        "${IMAGE_NAME}" \
        "${COMMAND}"
fi


