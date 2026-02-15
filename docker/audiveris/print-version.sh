#!/usr/bin/env bash
set -euo pipefail

help_out="$(Audiveris -help 2>&1 || true)"
version="$(printf '%s\n' "${help_out}" | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1 || true)"

if [[ -z "${version}" ]]; then
  echo "ERROR: unable to parse Audiveris version from -help output" >&2
  exit 1
fi

echo "${version}"
