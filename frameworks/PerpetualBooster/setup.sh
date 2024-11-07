#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/microsoft/FLAML.git"}
PKG=${3:-"flaml"}


. ${HERE}/../shared/setup.sh ${HERE} true

PIP install -y perpetual
