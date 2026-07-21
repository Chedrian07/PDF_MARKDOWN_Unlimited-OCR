# PaddleOCR-VL-1.6 CUDA sidecar — RTX 5070 Ti (Blackwell, sm_120) 대상
#
# 공식 NVIDIA Blackwell 가이드의 wheel 경로를 따른다
# (https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html):
#   1) paddlepaddle-gpu (공식 cu129 인덱스 — 가이드 최소 3.2.1, 여기서는 3.3.1 고정)
#   2) paddleocr[doc-parser] 3.6.x (PaddleOCR-VL-1.6 동봉 버전)
# 공식 sm120 Docker 이미지(ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:
# paddleocr3.6-nvidia-gpu-sm120 계열)도 대안이다 — 접근성이 낮아(중국 레지스트리)
# 재현 가능한 자체 빌드를 기본으로 한다 (docs/PADDLEOCR_VL_BLACKWELL_5070TI.md).
#
# 요구: 호스트 NVIDIA driver가 CUDA 12.9+ 지원, nvidia container toolkit.
# HF 토큰은 빌드 레이어에 넣지 않는다 — compose가 runtime env로만 전달.

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 libgl1 curl ccache \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/paddleocr_vl

# 1) PaddlePaddle GPU (공식 cu129 인덱스, 버전 고정 — latest 금지)
RUN pip install --no-cache-dir paddlepaddle-gpu==3.3.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# 2) PaddleOCR + sidecar 의존성 (PyPI, 버전 고정)
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf \
    PADDLE_PDX_MODEL_SOURCE=huggingface

EXPOSE 8080

# 최초 기동은 모델 다운로드(~1GB + layout 모델)가 포함될 수 있어 start-period를 길게
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=5 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
