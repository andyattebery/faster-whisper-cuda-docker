# Base image constraints, all three learned the hard way:
#
#   CUDA 12.x only - NOT 13. ctranslate2 dlopens libcublas.so.12; CUDA 13 images
#   ship libcublas.so.13, so a 13.x base builds fine and dies at model load.
#   (OpenNMT/CTranslate2#1933, still open.)
#
#   NOT the -cudnn variant. ctranslate2 stopped using cuDNN in 4.6.3, when the
#   default conv1d GPU path was rewritten in pure CUDA. The cuDNN runtime is
#   ~1GB, so shipping it would be dead weight. ci/smoke-test.py asserts the
#   library set to catch a future version reintroducing the dependency.
#
#   Host NVIDIA driver >= 570.124.06. ctranslate2 4.6.3 moved to CUDA Toolkit
#   12.8 and its wheels carry PTX that older drivers cannot JIT, failing with
#   "cudaErrorUnsupportedPtxVersion". This is stricter than the base image's own
#   NVIDIA_REQUIRE_CUDA (driver >= 535) and is the binding constraint.
#   (OpenNMT/CTranslate2#1978, SYSTRAN/faster-whisper#1413.)
ARG CUDA_VERSION='12.9.2'
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

# Install Whisper
WORKDIR /usr/src
ARG WYOMING_WHISPER_VERSION='3.6.0'
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ctranslate2 is installed before wyoming-faster-whisper on purpose: it decides
# the CUDA/cuDNN linkage, and faster-whisper's own '<5,>=4.0' constraint would
# otherwise permit 4.0-4.6.2, which need cuDNN that this base does not ship.
# The floor is 4.6.3 for that reason and for the driver requirement above.
#
# Deliberately NO `pip install -U setuptools wheel`: on Ubuntu 24.04 that fails
# with "Cannot uninstall wheel ... RECORD file not found" because apt owns those
# packages. It is also unnecessary - pip's PEP 517 build isolation fetches its
# own build deps when it builds the wyoming-faster-whisper sdist.
#
# netcat-openbsd is only for the HEALTHCHECK below.
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        netcat-openbsd \
    && pip3 install --no-cache-dir \
        'ctranslate2>=4.6.3,<5' \
        "wyoming-faster-whisper @ https://github.com/OHF-Voice/wyoming-faster-whisper/archive/refs/tags/v${WYOMING_WHISPER_VERSION}.tar.gz" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY run.sh ./

EXPOSE 10300

# Speaks the Wyoming protocol rather than just checking the port is bound: a
# wedged model load leaves the process up but unable to answer. --start-period
# covers the first-run model download.
HEALTHCHECK --start-period=10m --interval=1m --timeout=10s \
    CMD echo '{ "type": "describe" }' | nc -w 1 localhost 10300 | grep -q "faster-whisper" || exit 1

ENTRYPOINT ["bash", "/run.sh"]
