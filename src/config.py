"""경로·데이터셋 정의."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
REPORT = ROOT / "reports"
for p in (RAW, PROC, FIG, REPORT):
    p.mkdir(parents=True, exist_ok=True)

# 공공데이터포털 파일데이터 (모두 산림청 제공, 이용허락범위 제한 없음)
DATASETS = {
    "15121380": "산림청_산불통계데이터",
    "15121205": "산림청_산불상태별 이력",
    "15121208": "산림청_산불상황관제시스템 산불신고 유형별 위험지수",
    "15121381": "산림청_산불상황관제시스템 산불 단계별 동원력 이력",
    "15092027": "산림청 국립산림과학원_대형산불위험예보목록정보",
    "15144788": "산림청_산불상황관제시스템_진화대원대기장소",
    "15144784": "산림청_산불상황관제시스템_담수용사방댐",
    # 연계 데이터
    "15150429": "기후에너지환경부 국립환경과학원_기상청측정소 정보",  # ASOS 지점 위경도
}

# 분석 기간 (산불통계데이터 수록 범위 + 실효습도 계산용 여유)
WX_START, WX_END = "2021-11-01", "2025-11-30"

# 행정동 경계 (통계청 SGIS 기반, vuski/admdongkor 가공, 공공누리 1유형 + CC BY 4.0)
ADM_URL = ("https://raw.githubusercontent.com/vuski/admdongkor/master/"
           "ver20250401/HangJeongDong_ver20250401.geojson")


def kma_key() -> str:
    """기상청 API허브 인증키: 환경변수 KMA_AUTHKEY → ~/.kma_authkey 순."""
    k = os.environ.get("KMA_AUTHKEY", "").strip()
    if not k:
        f = Path.home() / ".kma_authkey"
        if f.exists():
            k = f.read_text(encoding="utf-8").strip()
    if not k:
        raise SystemExit("KMA_AUTHKEY 환경변수 또는 ~/.kma_authkey 파일이 필요합니다.")
    return k
