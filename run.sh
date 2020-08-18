#!/bin/bash
docker run --rm -it --init \
  --volume="$PWD:/app" \
hidden-challenges-mr:latest bash