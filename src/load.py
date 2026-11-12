"""원본 파일 읽기 + 기본 정제 (데이터셋별 1함수)."""
import re
from glob import glob

import numpy as np
import pandas as pd

from config import RAW

D = RAW / "datagokr"

SIDO_ABBR = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
    "경기도": "경기", "강원도": "강원", "강원특별자치도": "강원", "충청북도": "충북",
    "충청남도": "충남", "전라북도": "전북", "전북특별자치도": "전북", "전라남도": "전남",
    "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주", "전남광주": "전남",
    "전남광주통합특별시": "전남",
}


def _csv(pattern: str) -> pd.DataFrame:
    path = glob(str(D / pattern))[0]
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp949"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    from io import StringIO
    df = pd.read_csv(StringIO(txt), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    df.attrs["encoding"] = enc
    df.attrs["file"] = path
    return df


def key_emd(x) -> str:
    """'성거읍'/'성거' → '성거' (읍·면·동·리 접미사 제거)."""
    if not isinstance(x, str) or not x:
        return ""
    x = x.replace(" ", "")
    return re.sub(r"(읍|면|동|리|가)$", "", x) if len(x) > 1 else x


def key_sgg(x) -> str:
    """'천안시서북구'/'천안시 서북구'/'천안' → '천안' (첫 시·군·구 단위의 어간)."""
    if not isinstance(x, str) or not x:
        return ""
    x = x.replace(" ", "")
    m = re.match(r"^(.+?)(시|군)(?=.)", x)  # '천안시서북구' → '천안'
    if m:
        return m.group(1)
    return re.sub(r"(시|군|구)$", "", x) if len(x) > 1 else x


def parse_addr(a: str):
    """'충청남도 천안시서북구 성거읍' → ('충남', '천안시서북구', '성거읍')."""
    p = a.split()
    sido = SIDO_ABBR.get(p[0], p[0])
    rest = p[1:]
    while len(rest) > 1 and re.search(r"리$", rest[-1]):  # '금남면 용포리' → 리 단위 제거
        rest = rest[:-1]
    if sido == "세종":  # 세종은 시군구 없음: '세종특별자치시 금남면'
        return sido, "세종시", rest[-1] if rest else ""
    if rest and re.search(r"(읍|면|동|가|로)$", rest[-1]) and len(rest) >= 2:
        return sido, "".join(rest[:-1]), rest[-1]
    if len(rest) == 1 and re.search(r"(읍|면|동|가)$", rest[0]) and not re.search(r"(시|군|구)$", rest[0]):
        return sido, "", rest[0]
    return sido, "".join(rest), ""


# ---------------------------------------------------------------- 1. 산불통계데이터
def load_stats() -> pd.DataFrame:
    df = _csv("15121380_unzip/산불발생통계*.csv")
    def dt(prefix):
        y, m, d, t = (df[f"{prefix}_{k}"] for k in ("년", "월", "일", "시간"))
        return pd.to_datetime(y + "-" + m.str.zfill(2) + "-" + d.str.zfill(2) + " " + t, errors="coerce")
    out = pd.DataFrame({
        "occ": dt("발생일시"),
        "ext_raw": dt("진화종료시간"),
        "weekday_raw": df["발생일시_요일"],
        "office": df["발생장소_관서"],
        "sido": df["발생장소_시도"],
        "sgg_raw": df["발생장소_시군구"],
        "emd_raw": df["발생장소_읍면"],
        "ri_raw": df["발생장소_동리"],
        "cause_code": df["발생원인_구분"],
        "cause_detail": df["발생원인_세부원인"],
        "cause_etc": df["발생원인_기타"],
        "area_ha": pd.to_numeric(df["피해면적_합계"], errors="coerce"),
        "ext_year_raw": pd.to_numeric(df["진화종료시간_년"], errors="coerce"),
    })
    # 업무규칙 위배 플래그 (데이터사전에 정의돼 있으나 실제 파일엔 없는 '데이터오류_플래그'를 재현)
    flag = np.where(out.ext_year_raw > out.occ.dt.year + 1, "진화종료연도_이상",
                    np.where(out.ext_raw < out.occ, "진화종료시각_역전", ""))
    out["data_error_flag"] = flag
    out["ext"] = out.ext_raw.where(out.data_error_flag == "")
    out["dur_total_min"] = (out.ext - out.occ).dt.total_seconds() / 60
    out["date"] = out.occ.dt.normalize()
    out["sgg_k"] = out.sgg_raw.map(key_sgg)
    out["emd_k"] = out.emd_raw.map(key_emd)
    out["cause"] = out.apply(lambda r: normalize_cause(r.cause_detail, r.cause_etc), axis=1)
    out = out.sort_values("occ").reset_index(drop=True)
    out.insert(0, "fire_no", np.arange(1, len(out) + 1))
    return out


CAUSE_RULES = [  # (표준 원인, 키워드) — 위에서부터 우선
    ("방화", r"방화"),
    ("낙뢰", r"낙뢰"),
    ("성묘객실화", r"성묘"),
    ("논밭두렁소각", r"논|밭두렁|두렁"),
    ("농산폐기물소각", r"농산|영농|부산물|과수|폐비닐|농업|고춧대|깻대|전정|잔가지|영농부산물"),
    ("쓰레기소각", r"쓰레기|소각"),
    ("담뱃불실화", r"담뱃|담배"),
    ("입산자실화", r"입산|등산|산행|약초|임산물"),
    ("작업장실화", r"작업|용접|예초|공사|건설|그라인더|절단|벌목|산업|기계|차량|농기계|불티"),
    ("건축물화재비화", r"주택|건물|건축|화재비화|창고|비닐하우스|축사|공장|사찰|화목|보일러|아궁이"),
    ("전기·전선", r"전선|전기|전신주|스파크|변압기|송전|배전|단락"),
    ("어린이불장난", r"어린이|불장난|학생|청소년"),
    ("기타·미상", r"."),
]
STD_CAUSES = {"입산자실화", "논밭두렁소각", "쓰레기소각", "담뱃불실화", "성묘객실화",
              "어린이불장난", "작업장실화", "건축물화재비화", "농산폐기물소각", "방화", "낙뢰"}


def normalize_cause(detail, etc) -> str:
    if isinstance(detail, str) and detail in STD_CAUSES:
        return detail
    txt = etc if isinstance(etc, str) else ""
    if not txt or re.fullmatch(r"(원인)?미상|조사중|불명|확인중|-", txt):
        return "기타·미상"
    for name, pat in CAUSE_RULES:
        if re.search(pat, txt):
            return name
    return "기타·미상"


# ---------------------------------------------------------------- 2. 산불상태별 이력
def load_history() -> pd.DataFrame:
    df = _csv("15121205_*.csv")
    out = pd.DataFrame({
        "fire_id": df["산불정보아이디"].astype(int),
        "date_h": pd.to_datetime(df["산불신고일"]),
        "addr": df["산불발생주소"],
        "attack_start": pd.to_datetime(df["진화시작시간"], errors="coerce"),
        "attack_end": pd.to_datetime(df["진화완료시간"], errors="coerce"),
        "agency": df["관할기관명"],
        "sunrise": pd.to_timedelta(df["일출시간"]),
        "sunset": pd.to_timedelta(df["일몰시간"]),
    })
    parts = out.addr.map(parse_addr)
    out["sido_h"] = parts.str[0]
    out["sgg_h"] = parts.str[1]
    out["emd_h"] = parts.str[2]
    out["sgg_k_h"] = out.sgg_h.map(key_sgg)
    out["emd_k_h"] = out.emd_h.map(key_emd)
    return out


# ---------------------------------------------------------------- 3. 신고유형별 위험지수
def load_risk() -> pd.DataFrame:
    df = _csv("15121208_*.csv")
    return pd.DataFrame({
        "date_r": pd.to_datetime(df["발생일자"]),
        "addr": df["주소"],
        "report_type": df["신고유형"],
        "danger_index": pd.to_numeric(df["위험지수"], errors="coerce"),
        "sms_request": df["재난문자요청여부"],
    })


# ---------------------------------------------------------------- 4. 단계별 동원력 이력
def load_mobilization() -> pd.DataFrame:
    df = _csv("15121381_*.csv")
    out = pd.DataFrame({"date_m": pd.to_datetime(df["발생일자"]), "addr": df["주소"]})
    for lv in (1, 2, 3):
        out[f"mob{lv}_at"] = pd.to_datetime(df[f"산불{lv}단계발령일시"].replace("공란", np.nan), errors="coerce")
    out["mob_level"] = out[[f"mob{lv}_at" for lv in (1, 2, 3)]].notna().sum(axis=1)
    return out


# ---------------------------------------------------------------- 5. 대형산불위험예보목록
def load_forecast() -> pd.DataFrame:
    df = _csv("15092027_*.csv")
    out = pd.DataFrame({
        "t": pd.to_datetime(df["예보일시"]),
        "sido_raw": df["시도명"],
        "sgg": df["시군구명"],
        "emd": df["읍면동명"],
        "eff_hum": pd.to_numeric(df["실효습도"], errors="coerce"),
        "wind": pd.to_numeric(df["풍속"], errors="coerce"),
        "grade": df["등급"],
    })
    out["sido"] = out.sido_raw.map(lambda s: SIDO_ABBR.get(s, s))
    out["sgg_k"] = out.sgg.map(key_sgg)
    out["emd_k"] = out.emd.map(key_emd)
    return out


# ---------------------------------------------------------------- 6. 진화 자원 위치
def load_resources() -> dict:
    crew = _csv("15144788_*.csv")
    dam = _csv("15144784_*.csv")
    crew = crew.assign(lon=pd.to_numeric(crew["경도"]), lat=pd.to_numeric(crew["위도"]))
    crew = crew.drop_duplicates(["대기장소명", "lon", "lat"])
    heli = crew[crew["대기장소명"].str.contains("산림항공관리소")].drop_duplicates("대기장소명")
    return {
        "crew": crew[["관리기관", "대기장소명", "lon", "lat"]].reset_index(drop=True),
        "heli": heli[["대기장소명", "lon", "lat"]].reset_index(drop=True),
        "dam": dam.assign(lon=pd.to_numeric(dam["경도"]), lat=pd.to_numeric(dam["위도"]))[["주소", "lon", "lat"]],
    }
