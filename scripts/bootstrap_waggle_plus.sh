#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
TEMPLATE_DIR="$REPO_ROOT/templates/waggle-plus"

if [ "${1-}" = "" ] || [ "${1-}" = "--help" ] || [ "${1-}" = "-h" ]; then
    echo "Usage: scripts/bootstrap_waggle_plus.sh /path/to/private/waggle-plus"
    echo
    echo "Copies the public waggle_plus scaffold into a new target directory."
    exit 0
fi

TARGET_DIR=$1

if [ ! -d "$TEMPLATE_DIR" ]; then
    echo "Template directory not found: $TEMPLATE_DIR" >&2
    exit 1
fi

if [ -e "$TARGET_DIR" ]; then
    echo "Refusing to overwrite existing path: $TARGET_DIR" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"
cp -R "$TEMPLATE_DIR"/. "$TARGET_DIR"/

echo "Created private waggle_plus scaffold at: $TARGET_DIR"
echo
echo "Next steps:"
echo "  1. cd $TARGET_DIR"
echo "  2. git init"
echo "  3. pip install -e ."
echo "  4. python -m pytest tests/test_contract.py"
echo "  5. replace ExampleOIDCProvider with your real OIDC and RBAC logic"
