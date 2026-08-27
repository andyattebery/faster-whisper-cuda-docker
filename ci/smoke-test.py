"""Verify a built image from the inside. Run via:

    docker run --rm -i -e WANT_WHISPER=... -e WANT_CUDA=... \
        --entrypoint python3 <image> - < ci/smoke-test.py

CI has no NVIDIA hardware, so this cannot prove CUDA works. It proves the
things that are checkable without a GPU and that have actually broken before.
"""

import glob
import os
import re
import sys

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


print("== versions ==")

import ctranslate2  # noqa: E402
import faster_whisper  # noqa: E402
import wyoming_faster_whisper as wfw  # noqa: E402

# Floor is 4.6.3, not 4.5: that release both dropped the cuDNN dependency and
# raised the minimum NVIDIA driver to 570.124.06 (CUDA 12.8 PTX).
# See OpenNMT/CTranslate2#1978 and SYSTRAN/faster-whisper#1413.
ct2 = tuple(int(x) for x in ctranslate2.__version__.split(".")[:3])
check("ctranslate2 in [4.6.3, 5.0)", (4, 6, 3) <= ct2 < (5, 0, 0), ctranslate2.__version__)

want_whisper = os.environ["WANT_WHISPER"]
check(
    f"wyoming-faster-whisper == {want_whisper}",
    wfw.__version__ == want_whisper,
    wfw.__version__,
)

want_cuda = os.environ["WANT_CUDA"]
check(
    f"base image CUDA_VERSION == {want_cuda}",
    os.environ.get("CUDA_VERSION") == want_cuda,
    os.environ.get("CUDA_VERSION", "<unset>"),
)

print(f"  ---   faster-whisper {faster_whisper.__version__}")

print("== torch must NOT be present ==")
try:
    import torch
except ModuleNotFoundError:
    check("torch absent", True)
else:
    check("torch absent", False, f"found {torch.__version__}")

# The set of CUDA libraries ctranslate2 dlopens is an internal implementation
# detail, not a documented contract - it has changed in PATCH releases before
# (cuDNN dropped in 4.6.3). This image ships a base WITHOUT cuDNN, which is only
# safe while that set stays as below. Assert it so a change fails the build here
# rather than at model-load time on the GPU host.
print("== ctranslate2 external CUDA library set ==")

EXPECTED = {"libcublas.so.12", "libcuda.so.1", "libnccl.so.2"}
INTERESTING = ("cud", "cublas", "nccl", "nvrtc", "cufft", "cusparse", "curand", "cusolver", "npp")

pkg_dir = os.path.dirname(ctranslate2.__file__)
search_dirs = [pkg_dir, os.path.join(os.path.dirname(pkg_dir), "ctranslate2.libs")]

found = set()
scanned = 0
for d in search_dirs:
    for path in glob.glob(os.path.join(d, "**", "*.so*"), recursive=True):
        scanned += 1
        with open(path, "rb") as fh:
            blob = fh.read()
        for name in re.findall(rb"lib[a-zA-Z0-9_]+\.so(?:\.[0-9]+)*", blob):
            text = name.decode()
            if any(k in text for k in INTERESTING):
                found.add(text)

check("scanned at least one shared object", scanned > 0, f"{scanned} files")
check(
    "CUDA library set unchanged",
    found == EXPECTED,
    f"found {sorted(found)}",
)
if found != EXPECTED:
    print(f"        expected: {sorted(EXPECTED)}")
    print(f"        added:    {sorted(found - EXPECTED)}")
    print(f"        removed:  {sorted(EXPECTED - found)}")
    if any("cudnn" in x for x in found):
        print("        NOTE: cuDNN reappeared. The base image is the non-cudnn")
        print("              variant, so this WILL fail at model load on a GPU.")
        print("              Switch back to nvidia/cuda:<ver>-cudnn-runtime-*.")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
    sys.exit(1)
print("All checks passed.")
