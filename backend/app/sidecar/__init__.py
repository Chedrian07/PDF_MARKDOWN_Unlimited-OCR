"""sidecar 엔진 공통 계층: 프로토콜 스키마·HTTP 클라이언트·artifact materializer.

모델별 코드는 services/<engine>/ 컨테이너에 격리되고, 메인 backend에는
모델 독립적인 프로토콜 검증·파일 산출물 생성만 둔다 (docs/OCR_ENGINE_PROTOCOL.md).
"""
