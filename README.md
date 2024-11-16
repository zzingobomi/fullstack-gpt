# Fullstack-GTP

# 가상환경 설정

- python -m venv ./env
- \<venv\>/bin/Activate.bat (Windows)
- source \<venv\>/bin/activate (POSIX)

# Torch GPU Cuda 사용

- CUDA 툴킷 설치 (v11.8)
  - https://developer.nvidia.com/cuda-11-8-0-download-archive
- torch cuda 버전 설치
  - 다른 방법도 있는지 확인해볼것
  - pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
