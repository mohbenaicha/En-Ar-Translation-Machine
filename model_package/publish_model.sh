#!/bin/bash

# Building package and uploading it to PyPI

USER=$PYPI_USER
PASS=$PYPI_PASS

set -e

# assign current working directory
BASE_DIR=$(pwd)

# takes current directory as argument
DIRS="$@"

# target setup.py file
SETUP="setup.py"

# function for warnings, takes warning message following function call and directs it stdout to stderr
warn() {
    echo "$@" 1>&2
}

# kills terminal
die() {
    warn "$@"
    exit 1
}

build() {
    DIR="${1/%\//}"
    echo "Checking directory $DIR"
    cd "$BASE_DIR/$DIR"
    [ ! -e $SETUP ] && warn "No $SETUP file, skipping" && return
    
    # declare full name of package for warning before killing terminal
    PACKAGE_NAME=$(python $SETUP --fullname)
    echo "Found package $PACKAGE_NAME"
    
    # build source dist and wheel
    python "$SETUP" sdist bdist_wheel || warn "Building package $PACKAGE_NAME failed"
    
    # upload both sdist and wheel to PyPI

    for X in $(ls dist)
    do
        # twine to upload to PyPI
        twine upload "dist/$X" -u "$USER" -p "$PASS" || warn "Uploading package $PACKAGE_NAME failed on file dist/$X"
    done
}

if [ -n "$DIRS" ]; then
    for dir in $DIRS; do
        build $dir 
    done
# if [ -n "$DIRS" ]; then
#     ls -d */ | while read dir; do
#         build $dir
#     done
else
    echo "Attempting to build from $BASE_DIR.." && build
fi