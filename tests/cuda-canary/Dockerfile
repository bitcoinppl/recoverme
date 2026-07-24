FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749 AS build

ARG RUST_TOOLCHAIN=stable
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 --fail --silent --show-error \
        --output /tmp/rustup-init.sh \
        https://sh.rustup.rs \
    && sh /tmp/rustup-init.sh \
        -y \
        --default-toolchain "${RUST_TOOLCHAIN}" \
        --profile minimal \
        --no-modify-path \
    && rm /tmp/rustup-init.sh

WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY tests/cuda_canary.rs ./tests/cuda_canary.rs

RUN cargo test --locked --release --features cuda --test cuda_canary --no-run \
    && canary="$(find target/release/deps -maxdepth 1 -type f -name 'cuda_canary-*' -perm -0100)" \
    && test -n "${canary}" \
    && install -m 0755 "${canary}" /usr/local/bin/recoverme-cuda-canary

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749

COPY --from=build /usr/local/bin/recoverme-cuda-canary /usr/local/bin/recoverme-cuda-canary

ENTRYPOINT ["/usr/local/bin/recoverme-cuda-canary"]
CMD ["--ignored", "--exact", "cuda_public_fixture_matches_reference", "--nocapture", "--test-threads=1"]
