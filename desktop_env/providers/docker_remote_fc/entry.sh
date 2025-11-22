#!/usr/bin/env bash
set -Eeuo pipefail

APP="QEMU"
SUPPORT="https://github.com/qemus/qemu-docker"

cd /run

prepare_env(){
    . reset.sh      # Initialize system
    . install.sh    # Get bootdisk
    . disk.sh       # Initialize disks
    . display.sh    # Initialize graphics
    . network.sh    # Initialize network
    . boot.sh       # Configure boot
    . proc.sh       # Initialize processor
    . config.sh     # Configure arguments
    trap - ERR

    version=$(qemu-system-x86_64 --version | head -n 1 | cut -d '(' -f 1 | awk '{ print $NF }')
    info "Booting image${BOOT_DESC} using QEMU v$version..."
}

prepare_env > /tmp/prepare_env.log 2>&1

exec qemu-system-x86_64 ${ARGS:+ $ARGS} > /tmp/qemu.log 2>&1