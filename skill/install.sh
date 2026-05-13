#!/usr/bin/env bash
# Tokamak skill installer.
#
# Picks the prebuilt binary matching the host platform out of bin/, drops it
# at ~/.local/bin/tokamak, and (if needed) tells the user to add that path to
# their shell PATH. Idempotent — safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin"
INSTALL_DIR="${TOKAMAK_INSTALL_DIR:-${HOME}/.local/bin}"

osname="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"

case "${osname}-${arch}" in
  darwin-arm64)   target="tokamak-darwin-arm64" ;;
  darwin-x86_64)  target="tokamak-darwin-x86_64" ;;
  linux-x86_64)   target="tokamak-linux-x86_64" ;;
  linux-aarch64)  target="tokamak-linux-aarch64" ;;
  *)              echo "tokamak: no prebuilt for ${osname}/${arch}. Build from source: cd rust && cargo install --path tokamak" >&2; exit 1 ;;
esac

source_path="${BIN_DIR}/${target}"
if [ ! -x "${source_path}" ] && [ ! -f "${source_path}" ]; then
  echo "tokamak: binary missing: ${source_path}" >&2
  echo "tokamak: run \`gh release download --pattern '${target}' --dir '${BIN_DIR}'\` or build from source." >&2
  exit 1
fi

mkdir -p "${INSTALL_DIR}"
dest="${INSTALL_DIR}/tokamak"

# If a matching link already points at this binary, skip.
if [ -L "${dest}" ] && [ "$(readlink "${dest}")" = "${source_path}" ]; then
  echo "tokamak: already installed at ${dest}"
else
  ln -sf "${source_path}" "${dest}"
  chmod +x "${source_path}"
  echo "tokamak: installed at ${dest}"
fi

case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *) echo "tokamak: ${INSTALL_DIR} is not on PATH. Add it to your shell profile:"
     echo "  export PATH=\"${INSTALL_DIR}:\$PATH\"" ;;
esac

"${dest}" --version
