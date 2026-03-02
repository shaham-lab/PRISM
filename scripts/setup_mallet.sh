#!/usr/bin/env bash
set -euo pipefail

# ---- Configuration ----
MALLET_URL="https://mallet.cs.umass.edu/dist/mallet-2.0.8-source.tar.gz"
TAR_NAME="mallet-2.0.8-source.tar.gz"
EXTRACTED_DIR="mallet-2.0.8"
MALLET_DIR="mallet"

PATCH_SRC="beta_vector_patch/mallet"   # <-- in your repo
TARGET_DIR="${MALLET_DIR}/src/cc/mallet"

# ---- Checks ----
if [[ -d "${MALLET_DIR}" ]]; then
  echo "ERROR: './${MALLET_DIR}' already exists. Remove it (or rename it) and rerun."
  exit 1
fi

if [[ ! -d "${PATCH_SRC}" ]]; then
  echo "ERROR: Patch directory not found: '${PATCH_SRC}'"
  echo "Expected to find: beta_vector_patch/mallet"
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: 'curl' not found. Install curl and rerun."
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "ERROR: 'tar' not found."
  exit 1
fi

if ! command -v ant >/dev/null 2>&1; then
  echo "ERROR: 'ant' not found. Install Apache Ant and rerun."
  exit 1
fi

# ---- Step 1: Download + Extract + Rename ----
echo "==> Downloading MALLET source..."
curl -L -o "${TAR_NAME}" "${MALLET_URL}"

echo "==> Extracting..."
tar -xzf "${TAR_NAME}"

if [[ ! -d "${EXTRACTED_DIR}" ]]; then
  echo "ERROR: Expected extracted folder '${EXTRACTED_DIR}' not found after extraction."
  exit 1
fi

echo "==> Renaming '${EXTRACTED_DIR}' -> '${MALLET_DIR}'"
mv "${EXTRACTED_DIR}" "${MALLET_DIR}"

# ---- Step 2: Patch src/cc/mallet ----
if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "ERROR: Target directory not found: '${TARGET_DIR}'"
  exit 1
fi

echo "==> Backing up original MALLET directory: ${TARGET_DIR} -> ${TARGET_DIR}.ORIG"
mv "${TARGET_DIR}" "${TARGET_DIR}.ORIG"

echo "==> Copying patch into place: ${PATCH_SRC} -> ${TARGET_DIR}"
cp -R "${PATCH_SRC}" "${TARGET_DIR}"

# ---- Step 3: Build ----
echo "==> Building MALLET (ant clean && ant)..."
cd "${MALLET_DIR}"
ant clean
ant

echo
echo "Done."
echo "Patched MALLET is ready at: ./${MALLET_DIR}"
echo "Backup saved at: ./${MALLET_DIR}/src/cc/mallet.ORIG"
