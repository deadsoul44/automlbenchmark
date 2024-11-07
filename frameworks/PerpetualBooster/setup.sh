#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/perpetual-ml/perpetual.git"}
PKG=${3:-"perpetual"}


. ${HERE}/../shared/setup.sh ${HERE} true

PIP install perpetual
