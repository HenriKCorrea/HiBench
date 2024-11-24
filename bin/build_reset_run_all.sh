#!/bin/bash

set -u

# Set binaries directory
BIN_DIR="$(cd "$(dirname "$0")"; pwd)"

# # Set root directory
ROOT_DIR="$(cd "${BIN_DIR}/.."; pwd)"

# # Set docker opensource scripts directory
DOCKER_OPENSOURCE_SCRIPTS_DIR="${ROOT_DIR}/docker/opensource-docker/scripts"

# Build all benchmarks
echo ""
echo "=================="
echo "Building all benchmarks..."
echo "=================="
echo ""
"${BIN_DIR}/build_all.sh"

# Restart Hadoop and Spark services
echo ""
echo "=================="
echo "Restarting Hadoop and Spark services..."
echo "=================="
echo ""
"${DOCKER_OPENSOURCE_SCRIPTS_DIR}/restart_hadoop_spark.sh"

# Run all benchmarks
echo ""
echo "=================="
echo "Running all benchmarks..."
echo "=================="
echo ""
"${BIN_DIR}/run_all.sh"
