#!/bin/sh
set -eu

usage() {
    echo "usage: $0 <image-tag>" >&2
    exit 2
}

find_command() {
    command -v "$1" 2>/dev/null && return
    for directory in /opt/homebrew/bin /usr/local/bin; do
        if test -x "${directory}/$1"; then
            echo "${directory}/$1"
            return
        fi
    done
    return 1
}

test "$#" -eq 1 || usage

image=$1
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH='' cd -- "${script_dir}/../.." && pwd)
docker_command=$(find_command docker || true)
nsc_command=$(find_command nsc || true)
skopeo_command=$(find_command skopeo || true)
crane_command=$(find_command crane || true)

cd "${repository_root}"

if test -n "${docker_command}" && "${docker_command}" buildx version >/dev/null 2>&1; then
    "${docker_command}" buildx build \
        --platform linux/amd64 \
        --file tests/cuda-canary/Dockerfile \
        --tag "${image}" \
        --push \
        .
elif test -n "${nsc_command}"; then
    "${nsc_command}" build \
        --platform linux/amd64 \
        --file tests/cuda-canary/Dockerfile \
        --tag "${image}" \
        --push \
        .
else
    echo "Docker Buildx or nsc is required to build the CUDA canary image" >&2
    exit 127
fi

if test -n "${skopeo_command}"; then
    digest=$("${skopeo_command}" inspect --format '{{.Digest}}' "docker://${image}")
elif test -n "${crane_command}"; then
    digest=$("${crane_command}" digest "${image}")
else
    echo "skopeo or crane is required to resolve the pushed image digest" >&2
    exit 127
fi

echo "Built and pushed: ${image}"
echo "Set gpuq.toml image to ${image}@${digest}"
