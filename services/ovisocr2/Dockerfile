# OvisOCR2 CUDA sidecar — RTX 5070 Ti (Blackwell, sm_120) 대상
#
# 베이스: vLLM 공식 이미지 v0.22.1 (모델 카드 공식 권장 버전), cu129 태그.
#   - Blackwell은 CUDA >= 12.8 필요 — cu129 태그가 sm_120 커널을 포함한다.
#   - 이미지에 torch(cu129)·vllm==0.22.1·fastapi·uvicorn·pillow가 동봉되어 있어
#     추가 의존성은 requirements.lock의 최소 항목뿐이다.
#   - `latest` 태그 금지 — 버전 고정 (docs/OVISOCR2_CUDA_5070TI.md).
#
# HF 토큰은 빌드 레이어에 절대 넣지 않는다 — compose가 runtime env로만 전달.

FROM vllm/vllm-openai:v0.22.1-cu129

WORKDIR /srv/ovisocr2

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf

EXPOSE 8080

# 최초 기동은 모델 다운로드(~1.7GB)+로드가 포함될 수 있어 start-period를 길게 둔다.
# health는 로드 전에도 200(model_loaded=false)을 반환하므로 여기서는 연결성만 본다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=5 \
    CMD python3 -c "import urllib.request,sys; r=urllib.request.urlopen('http://localhost:8080/health', timeout=4); sys.exit(0 if r.status==200 else 1)" || exit 1

# 베이스 이미지의 OpenAI 서버 entrypoint를 비우고 sidecar 앱으로 대체
ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
