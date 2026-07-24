#!/bin/sh
set -eu

usage() {
    echo "usage: $0 <load|push> <image-tag>" >&2
    exit 2
}

test "$#" -eq 2 || usage

case "$1" in
    load)
        output_flag=--load
        ;;
    push)
        output_flag=--push
        ;;
    *)
        usage
        ;;
esac

image=$2
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "${script_dir}/../.." && pwd)

cd "${repository_root}"
exec docker buildx build \
    --platform linux/amd64 \
    --file tests/cuda-canary/Dockerfile \
    --tag "${image}" \
    "${output_flag}" \
    .
