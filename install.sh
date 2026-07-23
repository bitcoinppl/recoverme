#!/bin/sh

set -eu

repository=bitcoinppl/recoverme
version=${RECOVERME_VERSION:-v0.2.0}
install_dir=${RECOVERME_INSTALL_DIR:-/usr/local/bin}

fail() {
    printf 'recoverme installer: %s\n' "$*" >&2
    exit 1
}

printf '%s\n' "$version" |
    grep -Eq '^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$' ||
    fail "invalid version: $version"

system=$(uname -s)
machine=$(uname -m)
case "$system-$machine" in
    Darwin-arm64 | Darwin-aarch64)
        target=aarch64-apple-darwin
        ;;
    Linux-x86_64 | Linux-amd64)
        target=x86_64-unknown-linux-musl
        ;;
    Linux-aarch64 | Linux-arm64)
        target=aarch64-unknown-linux-musl
        ;;
    Darwin-*)
        fail "Intel macOS is not supported"
        ;;
    Linux-*)
        fail "unsupported Linux architecture: $machine"
        ;;
    *)
        fail "unsupported platform: $system-$machine"
        ;;
esac

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"
command -v install >/dev/null 2>&1 || fail "install is required"

archive="recoverme-$version-$target.tar.gz"
archive_dir=${archive%.tar.gz}
download_root="https://github.com/$repository/releases/download/$version"
temp_dir=$(mktemp -d) || fail "could not create a temporary directory"

cleanup() {
    rm -rf "$temp_dir"
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

download() {
    name=$1
    curl \
        --proto '=https' \
        --tlsv1.2 \
        --fail \
        --location \
        --silent \
        --show-error \
        --output "$temp_dir/$name" \
        "$download_root/$name"
}

printf 'Downloading recoverme %s for %s\n' "$version" "$target"
download "$archive"
download SHA256SUMS

awk -v archive="$archive" '
    $2 == archive {
        print
        found = 1
    }
    END {
        if (!found) {
            exit 1
        }
    }
' "$temp_dir/SHA256SUMS" >"$temp_dir/SELECTED_SHA256SUM" ||
    fail "SHA256SUMS does not contain $archive"

if command -v sha256sum >/dev/null 2>&1; then
    (cd "$temp_dir" && sha256sum --check SELECTED_SHA256SUM) ||
        fail "checksum verification failed"
elif command -v shasum >/dev/null 2>&1; then
    (cd "$temp_dir" && shasum -a 256 --check SELECTED_SHA256SUM) ||
        fail "checksum verification failed"
else
    fail "sha256sum or shasum is required"
fi

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    printf 'Verifying GitHub artifact attestations\n'
    (cd "$temp_dir" &&
        gh attestation verify SHA256SUMS --repo "$repository" &&
        gh attestation verify "$archive" --repo "$repository") ||
        fail "artifact attestation verification failed"
else
    printf 'Skipping artifact attestations; authenticated GitHub CLI not found\n'
fi

expected_entries=$(printf '%s\n' \
    "$archive_dir" \
    "$archive_dir/LICENSE-APACHE" \
    "$archive_dir/LICENSE-MIT" \
    "$archive_dir/README.md" \
    "$archive_dir/recoverme")
actual_entries=$(
    tar -tzf "$temp_dir/$archive" |
        sed 's:/$::' |
        LC_ALL=C sort
) || fail "could not inspect release archive"
[ "$actual_entries" = "$expected_entries" ] ||
    fail "release archive has unexpected contents"

tar -xzf "$temp_dir/$archive" -C "$temp_dir"
binary="$temp_dir/$archive_dir/recoverme"
if [ ! -f "$binary" ] || [ -L "$binary" ] || [ ! -x "$binary" ]; then
    fail "archive does not contain an executable recoverme binary"
fi

destination="$install_dir/recoverme"
if [ -d "$install_dir" ] && [ -w "$install_dir" ]; then
    install -m 0755 "$binary" "$destination"
elif [ ! -e "$install_dir" ] && [ -w "$(dirname "$install_dir")" ]; then
    mkdir -p "$install_dir"
    install -m 0755 "$binary" "$destination"
else
    command -v sudo >/dev/null 2>&1 ||
        fail "$install_dir is not writable and sudo is unavailable"
    sudo install -d -m 0755 "$install_dir"
    sudo install -m 0755 "$binary" "$destination"
fi

printf 'Installed recoverme %s to %s\n' "$version" "$destination"
case ":$PATH:" in
    *":$install_dir:"*) ;;
    *) printf 'Add %s to PATH before running recoverme\n' "$install_dir" ;;
esac
