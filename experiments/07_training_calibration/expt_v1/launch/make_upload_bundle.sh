#!/usr/bin/env bash
# Build the upload bundle for a cluster without a direct copy route: this repository at HEAD (committed files only)
# plus the frozen prepared data, laid out as zipmix_exp07/{repo,data}. Unpack it with `tar xzf <bundle> -C <parent
# of ZIPMIX_WORK>`. Refuses data whose manifests differ from the committed ones.
#   launch/make_upload_bundle.sh <prepared data dir with calibration/ and stage_b/> <out.tgz>
set -euo pipefail
data=$(cd -- "${1:?usage: make_upload_bundle.sh <data dir> <out.tgz>}" && pwd)
out=${2:?usage: make_upload_bundle.sh <data dir> <out.tgz>}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root=$(git -C "$here" rev-parse --show-toplevel)
for stage in calibration stage_b; do
  cmp -s "$data/$stage/manifest.json" "$here/${stage}_manifest.json" || { echo "$stage manifest differs from the committed one" >&2; exit 1; }
done
[[ -z $(git -C "$root" status --porcelain) ]] || { echo "commit first: the bundle holds HEAD only" >&2; exit 1; }
stage_dir=$(mktemp -d)
trap 'rm -rf "$stage_dir"' EXIT
mkdir -p "$stage_dir/zipmix_exp07/repo"
git -C "$root" archive HEAD | tar x -C "$stage_dir/zipmix_exp07/repo"
git -C "$root" rev-parse --short HEAD >"$stage_dir/zipmix_exp07/repo/COMMIT"
cp -r "$data" "$stage_dir/zipmix_exp07/data"
tar czf "$out" -C "$stage_dir" --owner=0 --group=0 zipmix_exp07
sha256sum "$out"
ls -l "$out"
