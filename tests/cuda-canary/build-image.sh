#!/bin/sh
set -eu

usage() {
    echo "usage: $0 <image-tag>" >&2
    exit 2
}

test "$#" -eq 1 || usage

image=$1
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH='' cd -- "${script_dir}/../.." && pwd)

if ! command -v skopeo >/dev/null 2>&1; then
    echo "skopeo is required to resolve the pushed image digest" >&2
    exit 127
fi

cd "${repository_root}"

if command -v docker >/dev/null 2>&1 && docker buildx version >/dev/null 2>&1; then
    docker buildx build \
        --platform linux/amd64 \
        --file tests/cuda-canary/Dockerfile \
        --tag "${image}" \
        --push \
        .
elif command -v nsc >/dev/null 2>&1; then
    nsc build \
        --platform linux/amd64 \
        --file tests/cuda-canary/Dockerfile \
        --tag "${image}" \
        --push \
        .
else
    echo "Docker Buildx or nsc is required to build the CUDA canary image" >&2
    exit 127
fi

digest=$(skopeo inspect --format '{{.Digest}}' "docker://${image}")

echo "Built and pushed: ${image}"
echo "Set gpuq.toml image to ${image}@${digest}"
