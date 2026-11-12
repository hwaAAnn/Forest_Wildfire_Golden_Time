"""활용 중 확인한 데이터 품질 이슈를 '재현 가능한 숫자'로 정리 → reports/data_quality.md

개선 아이디어(공모 19~21번 문항)의 근거. 모든 수치는 원본 파일에서 직접 계산한다.
"""
import json
import re
from glob import glob

import numpy as np
import pandas as pd

from config import PROC, RAW, REPORT
from load import STD_CAUSES, _csv, load_forecast, load_history, load_stats, normalize_cause

D = RAW / "datagokr"
# 공공데이터포털 상세 페이지 메타데이터 (2026-11-13 확인값)
PORTAL_META = {
    "15121380": {"temporalCoverage": "2022년 ~ 2024년 9월", "전체 행": "1", "업데이트 주기": "연간"},
}


def main():
    Q = {}
    # ------------------------------------------------ A. 산불통계데이터
    raw = _csv("15121380_unzip/산불발생통계*.csv")
    dic = pd.read_csv(glob(str(D / "15121380_unzip" / "*데이터사전*.csv"))[0], encoding="utf-8-sig")
    st = load_stats()
    dict_cols = dic["CSV헤더(한글)"].str.strip().tolist()
    missing_cols = [c for c in dict_cols if c not in raw.columns]
    Q["stats"] = dict(
        encoding=raw.attrs["encoding"], rows=len(raw), file_cols=raw.shape[1], dict_cols=len(dict_cols),
        dict_cols_missing_in_file=missing_cols,
        occ_min=str(st.occ.min()), occ_max=str(st.occ.max()),
        portal_meta=PORTAL_META["15121380"],
        err_reversed=int((st.data_error_flag == "진화종료시각_역전").sum()),
        err_year=int((st.data_error_flag == "진화종료연도_이상").sum()),
        err_year_values=sorted(st.loc[st.data_error_flag == "진화종료연도_이상", "ext_year_raw"].astype(int).unique().tolist()),
        cause_code_missing=float(raw["발생원인_구분"].isna().mean()),
        cause_etc_direct=int((raw["발생원인_세부원인"] == "기타(직접입력)").sum()),
        emd_missing=float(raw["발생장소_읍면"].isna().mean()),
        sgg_missing=float(raw["발생장소_시군구"].isna().mean()),
    )
    etc = raw[raw["발생원인_세부원인"] == "기타(직접입력)"]
    recl = etc.apply(lambda r: normalize_cause("기타(직접입력)", r["발생원인_기타"]), axis=1)
    Q["stats"]["cause_etc_reclassifiable"] = int(recl.isin(STD_CAUSES).sum())
    Q["stats"]["cause_etc_examples"] = etc["발생원인_기타"].value_counts().head(8).to_dict()
    Q["stats"]["cause_etc_unique_texts"] = int(etc["발생원인_기타"].nunique())

    # ------------------------------------------------ B. 산불상태별 이력
    hraw = _csv("15121205_*.csv")
    hi = load_history()
    start_day_diff = (hi.attack_start.dt.normalize() - hi.date_h).dt.days
    sun_per_day = hraw.groupby("산불신고일")["일몰시간"].nunique()
    fires = pd.read_csv(PROC / "fires.csv")
    ss_kfs = pd.to_timedelta(fires.sunset_kfs, errors="coerce")
    ss_loc = pd.to_timedelta(fires.sunset, errors="coerce")
    dev = (ss_kfs - ss_loc).dt.total_seconds().abs() / 60
    Q["history"] = dict(
        rows=len(hraw), date_min=str(hi.date_h.min().date()), date_max=str(hi.date_h.max().date()),
        start_not_on_report_day=int((start_day_diff != 0).sum()),
        start_after_report_max_days=int(start_day_diff.max()),
        end_before_start=int((hi.attack_end < hi.attack_start).sum()),
        days_with_single_sunset=float((sun_per_day == 1).mean()), n_days=int(len(sun_per_day)),
        sunset_dev_median_min=float(dev.median()), sunset_dev_max_min=float(dev.max()),
    )

    # ------------------------------------------------ C. 신고유형별 위험지수
    rraw = pd.read_csv(glob(str(D / "15121208_*.csv"))[0], encoding="cp949", dtype=str)
    rraw.columns = [c.strip() for c in rraw.columns]
    padded = {c: int((rraw[c].dropna() != rraw[c].dropna().str.strip()).sum()) for c in ["신고유형", "재난문자요청여부"]}
    di = pd.to_numeric(rraw["위험지수"], errors="coerce")
    by_size = fires.assign(miss=fires.danger_index.isna()).groupby(fires.area_ha >= 10).miss.mean()
    Q["risk"] = dict(
        rows=len(rraw), date_max=rraw["발생일자"].max(), missing=float(di.isna().mean()),
        missing_by_year=di.isna().groupby(rraw["발생일자"].str[:4]).mean().round(3).to_dict(),
        missing_small=float(by_size.get(False)), missing_10ha=float(by_size.get(True)),
        values_with_padding=padded,
    )
    # 2025년 대형산불의 신고 시점 위험지수 vs 같은 날 다른 산불
    f = fires.assign(date=pd.to_datetime(fires.date))
    rows = []
    for _, r in f[(f.year == 2025) & (f.area_ha >= 1000)].iterrows():
        same = f[(f.date == r.date) & (f.fire_no != r.fire_no)].danger_index
        rows.append(dict(addr=r.addr, area_ha=r.area_ha, index=r.danger_index, same_day_median=float(same.median())))
    Q["risk"]["mega_2025"] = rows

    # ------------------------------------------------ D. 데이터셋 간 결합
    d = fires
    occ, ext, aend = pd.to_datetime(d.occ), pd.to_datetime(d.ext_best), pd.to_datetime(d.attack_end)
    gap_h = (ext - aend).dt.total_seconds().abs() / 3600
    Q["join"] = dict(
        common_id=False, matched=int(d.fire_id.notna().sum()), total=len(d),
        end_mismatch_gt6h=int((gap_h > 6).sum()), end_mismatch_gt24h=int((gap_h > 24).sum()),
        compared=int(gap_h.notna().sum()),
        coverage_end={"산불통계데이터": Q["stats"]["occ_max"][:10], "산불상태별 이력": Q["history"]["date_max"],
                      "신고유형별 위험지수": Q["risk"]["date_max"],
                      "단계별 동원력 이력": _csv("15121381_*.csv")["발생일자"].max()},
    )

    # ------------------------------------------------ E. 대형산불위험예보목록
    fc = load_forecast()
    meets_warn = (fc.eff_hum < 30) & (fc.wind >= 11)
    Q["forecast"] = dict(
        rows=len(fc), t_min=str(fc.t.min()), t_max=str(fc.t.max()),
        sido_variants=int(fc.sido_raw.nunique()), sido_true=int(fc.sido.nunique()),
        sido_examples=sorted(fc.sido_raw.unique().tolist()),
        grade_counts=fc.grade.value_counts().to_dict(),
        meets_warn_but_jueuibo=int((meets_warn & (fc.grade == "주의보")).sum()),
        the_only_warn=fc[fc.grade == "경보"][["t", "sgg", "emd", "eff_hum", "wind"]].astype(str).to_dict("records"),
        wind_ge_30=int((fc.wind >= 30).sum()), wind_max=float(fc.wind.max()),
        effhum_zero=int((fc.eff_hum <= 0).sum()),
    )

    # ------------------------------------------------ F. 진화자원
    crew = _csv("15144788_*.csv")
    Q["resources"] = dict(crew_rows=len(crew), crew_dup_rows=int(crew.duplicated(["대기장소명", "경도", "위도"]).sum()))

    json.dump(Q, open(PROC / "quality_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    write_md(Q)
    print(json.dumps(Q, ensure_ascii=False, indent=1, default=str)[:6000])


def pct(x):
    return f"{100 * x:.1f}%"


def write_md(Q):
    s, h, r, j, f = Q["stats"], Q["history"], Q["risk"], Q["join"], Q["forecast"]
    lines = [
        "# 데이터 품질 점검 (활용 중 발견한 이슈)",
        "",
        "`python src/quality.py`로 원본 파일에서 다시 계산할 수 있다. 개선 아이디어의 근거 자료.",
        "",
        "| # | 데이터 | 확인한 사실 | 분석에 준 영향 |",
        "|---|---|---|---|",
        f"| 1 | 공통 | 산불통계·위험지수·동원령 데이터에 **공통 산불 ID가 없음** (이력 데이터에만 `산불정보아이디`) | 일자·지명·시각으로 추정 매칭: {j['matched']:,}/{j['total']:,}건({pct(j['matched'] / j['total'])}) |",
        f"| 2 | 산불통계 | 데이터사전에는 {s['dict_cols']}개 컬럼, 실제 CSV는 {s['file_cols']}개. 사전에 정의된 `표준_법정동코드`·`발생일시_결합`·`데이터오류_플래그` 등 {len(s['dict_cols_missing_in_file'])}개가 파일에 없음 | 행정코드·좌표 없이 지명 문자열로 지오코딩 |",
        f"| 3 | 산불통계 | 포털 메타데이터 수록기간 '{s['portal_meta']['temporalCoverage']}', 전체 행 '{s['portal_meta']['전체 행']}' ↔ 실제 {s['occ_min'][:10]}~{s['occ_max'][:10]}, {s['rows']:,}행 | 검색·선택 단계에서 오해 소지 |",
        f"| 4 | 산불통계 | 진화종료시각이 발생보다 이른 행 {s['err_reversed']}건, 진화종료연도 {s['err_year_values']} 등 이상 {s['err_year']}건 | 해당 행 지속시간 제외 |",
        f"| 5 | 산불통계 | 세부원인 '기타(직접입력)' {s['cause_etc_direct']:,}건 중 {s['cause_etc_reclassifiable']:,}건이 표준 원인(쓰레기소각, 성묘객실화 등)으로 재분류 가능. 자유기재 문구 {s['cause_etc_unique_texts']}종 | 키워드 규칙으로 원인 표준화 |",
        f"| 6 | 산불통계·이력 | 통계의 '진화종료'와 이력의 '진화완료'가 6시간 넘게 다른 산불 {j['end_mismatch_gt6h']}건(24시간 초과 {j['end_mismatch_gt24h']}건) | '24시간 이상 지속' 대형산불 기준을 쓸 수 없어 면적 기준만 사용 |",
        f"| 7 | 이력 | 진화시작이 신고일과 다른 날인 행 {h['start_not_on_report_day']}건(최대 {h['start_after_report_max_days']}일 뒤), 진화완료가 진화시작보다 이른 행 {h['end_before_start']}건 | 초기 대응 시간 일부 제외 |",
        f"| 8 | 이력 | 일출·일몰 시각이 같은 날이면 전국 동일 값(신고일의 {pct(h['days_with_single_sunset'])}). 위치별 계산값과 최대 {h['sunset_dev_max_min']:.0f}분 차이 | 위경도로 일몰 재계산 |",
        f"| 9 | 위험지수 | 위험지수 결측 {pct(r['missing'])} (연도별 {', '.join(f'{k}년 {pct(v)}' for k, v in r['missing_by_year'].items())}); 10ha 이상 산불 {pct(r['missing_10ha'])} vs 미만 {pct(r['missing_small'])} | 결측이 결과와 연관 → 당일 전국 중앙값으로 대체 |",
        f"| 10 | 위험지수 | 2025년 초대형산불의 신고 시점 지수가 같은 날 다른 산불보다 크게 낮음: " + "; ".join(f"{x['addr'].split()[1]} {x['addr'].split()[-1]} {x['index']:.0f} (같은 날 중앙값 {x['same_day_median']:.0f})" for x in r["mega_2025"]) + " | 산출 시각·격자·구성요소가 없어 원인 확인 불가 |",
        f"| 11 | 위험지수 | 값 뒤 공백 패딩: 신고유형 {r['values_with_padding']['신고유형']}건('수동신고 '), 재난문자요청여부 {r['values_with_padding']['재난문자요청여부']}건('N       ') | 정제 필요 |",
        f"| 12 | 대형산불위험예보 | 17개 시도가 **{f['sido_variants']}가지 표기**로 혼재 (예: 강원/강원도/강원특별자치도, 전남/전라남도/전남광주) | 시도명 정규화 사전 필요 |",
        f"| 13 | 대형산불위험예보 | 설명서의 경보 기준(실효습도 30% 미만·풍속 11m/s 이상)을 만족하는 {f['meets_warn_but_jueuibo']}건이 '주의보'로 기록. 전체 {f['rows']:,}건 중 '경보'는 1건이며 그 1건은 실효습도 {f['the_only_warn'][0]['eff_hum']}% | 등급을 그대로 쓰기 어려움 → '발령 여부'만 사용 |",
        f"| 14 | 대형산불위험예보 | 풍속 30m/s 이상 {f['wind_ge_30']}건(최대 {f['wind_max']}m/s), 실효습도 0% {f['effhum_zero']}건 | 이상치 점검 필요 |",
        f"| 15 | 공통 | 갱신 시점이 데이터마다 다름(수록 종료일: " + ", ".join(f"{k} {v}" for k, v in j["coverage_end"].items()) + ") | 결합 가능한 기간이 가장 짧은 데이터에 맞춰짐 |",
        f"| 16 | 공통 | 모든 CSV가 CP949 인코딩, zip 안 파일명도 CP949 | UTF-8 환경에서 깨짐 |",
        f"| 17 | 진화대원대기장소 | {Q['resources']['crew_rows']}행 중 같은 장소·좌표 중복 {Q['resources']['crew_dup_rows']}행 | 중복 제거 |",
    ]
    (REPORT / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
