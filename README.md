**A simple, convenient and transparent way to run Wyoming Faster Whisper using a CUDA compatible nvidia GPU.**


**Pull a prebuilt image**

```
docker pull ghcr.io/andyattebery/faster-whisper-cuda-docker:latest
```

Tags: `latest`, `3.6.0` (the wyoming-faster-whisper version), `3.6.0-cuda12.9.2` (fully qualified),
and `sha-<short>` for pinning an exact build.

**Or build it yourself**

```
git clone https://github.com/andyattebery/faster-whisper-cuda-docker
cd faster-whisper-cuda-docker
docker compose up -d
```

Edit the `volumes:` path in `docker-compose.yaml` first — it ships with a placeholder.

**Requirements**

- **NVIDIA driver >= 570.124.06.** This is the one most likely to bite you, and it is stricter than
  the CUDA base image asks for. `ctranslate2` 4.6.3 moved to CUDA Toolkit 12.8, and its wheels carry
  PTX that older drivers cannot JIT — you get `cudaErrorUnsupportedPtxVersion` at inference time, not
  at startup. Check with `nvidia-smi`.
  ([OpenNMT/CTranslate2#1978](https://github.com/OpenNMT/CTranslate2/issues/1978),
  [SYSTRAN/faster-whisper#1413](https://github.com/SYSTRAN/faster-whisper/issues/1413))
- **linux/amd64 with an NVIDIA GPU.** CTranslate2's arm64 wheels are built without CUDA, so this does
  not run on Apple Silicon or a Jetson. For Jetson, use
  [jetson-containers](https://github.com/dusty-nv/jetson-containers).
- The NVIDIA Container Toolkit, so the container can see the GPU.

**Why these versions**

- **CUDA 12.x, not CUDA 13.** CTranslate2 `dlopen`s `libcublas.so.12`, and CUDA 13 base images ship
  `libcublas.so.13`. A CUDA 13 base builds without complaint and then fails at model load.
  ([OpenNMT/CTranslate2#1933](https://github.com/OpenNMT/CTranslate2/issues/1933))
- **No cuDNN, on purpose.** CTranslate2 needed cuDNN only up to 4.6.2; 4.6.3 rewrote the conv1d GPU
  path in pure CUDA and dropped it. This image therefore uses the plain `-runtime` base rather than
  `-cudnn-runtime`, which saves about 1 GB. `ctranslate2>=4.6.3` is pinned to keep that true, and
  `ci/smoke-test.py` asserts the exact set of CUDA libraries the wheel references so a future release
  that reintroduces cuDNN fails the build instead of failing on the GPU host.

**Check if GPU is being used:**

`nvidia-smi`

You should see something like this:

![image](https://github.com/Cheerpipe/faster-whisper-cuda-docker/assets/972765/98cd9518-5044-469d-96b0-d0d083044831)

 
