"""데이터셋 결합 → 산불 1건 = 1행 분석 테이블 (data/processed/fires.csv).

결합 순서
  산불통계(2,020건) ─(일자·시도·시군구·읍면·시각 근접)→ 산불상태별 이력(진화착수·완료, 일출·일몰)
                    ─(일자·주소·순번)→ 신고유형별 위험지수
                    ─(일자·주소)→ 단계별 동원령
                    ─(행정동 경계 중심점)→ 위경도 → 최근접 산림항공관리소·진화대 대기장소·담수용 사방댐
                    ─(최근접 ASOS)→ 기상(실효습도·최대풍속·무강수일수)
                    ─(시군구·일시)→ 대형산불위험예보(주의보) 발령 여부
                    + 동시 진행 산불 수, 공휴일, 일몰까지 남은 시간
"""
import json
import math
import re
from glob import glob

import holidays
import numpy as np
import pandas as pd
from shapely.geometry import shape

from config import RAW, PROC
from load import (SIDO_ABBR, key_emd, key_sgg, load_forecast, load_history,
                  load_mobilization, load_resources, load_risk, load_stats)


# ---------------------------------------------------------------- 매칭: 통계 ↔ 이력
def match_stats_history(st: pd.DataFrame, hi: pd.DataFrame) -> pd.DataFrame:
    """후보(같은 날 또는 ±1일) 중 지명 일치 + 시각 근접 점수로 1:1 탐욕 매칭."""
    cands = []
    for shift in (0, 1, -1):
        h = hi.assign(_d=hi.date_h - pd.Timedelta(days=shift))
        m = st[["fire_no", "date", "occ", "sido", "sgg_k", "emd_k"]].merge(
            h[["fire_id", "_d", "attack_start", "sido_h", "sgg_k_h", "emd_k_h"]],
            left_on="date", right_on="_d")
        m["shift"] = shift
        cands.append(m)
    c = pd.concat(cands, ignore_index=True)
    c["gap"] = (c.attack_start - c.occ).dt.total_seconds() / 60
    c["emd_eq"] = (c.emd_k != "") & (c.emd_k == c.emd_k_h)
    c["sgg_eq"] = (c.sgg_k != "") & ((c.sgg_k == c.sgg_k_h) | c.apply(
        lambda r: bool(r.sgg_k) and bool(r.sgg_k_h) and (r.sgg_k in r.sgg_k_h or r.sgg_k_h in r.sgg_k), axis=1))
    c["sido_eq"] = c.sido == c.sido_h
    c = c[(c.emd_eq | c.sgg_eq)]
    # 진화착수는 발생 '이후'가 정상. 역전·장시간 격차는 감점
    gap_pen = np.where(c.gap.between(-15, 24 * 60), np.abs(c.gap) / 600, 5 + np.abs(c.gap) / 600)
    c["score"] = 3 * c.emd_eq + 2 * c.sgg_eq + 1 * c.sido_eq - gap_pen - 0.5 * (c["shift"] != 0)
    c = c.sort_values("score", ascending=False)
    used_s, used_h, rows = set(), set(), []
    for r in c.itertuples():
        if r.fire_no in used_s or r.fire_id in used_h:
            continue
        used_s.add(r.fire_no)
        used_h.add(r.fire_id)
        rows.append((r.fire_no, r.fire_id, r.score, r.gap))
    return pd.DataFrame(rows, columns=["fire_no", "fire_id", "match_score", "attack_gap_raw"])


def attach_risk(hi: pd.DataFrame, ri: pd.DataFrame) -> pd.DataFrame:
    """이력·위험지수는 주소 표기가 같음 → (일자, 주소, 일자·주소 내 순번)으로 결합."""
    h = hi.sort_values(["date_h", "fire_id"]).copy()
    h["k"] = h.groupby(["date_h", "addr"]).cumcount()
    r = ri.copy()
    r["k"] = r.groupby(["date_r", "addr"]).cumcount()
    return h.merge(r, left_on=["date_h", "addr", "k"], right_on=["date_r", "addr", "k"], how="left").drop(columns=["k", "date_r"])


def attach_mob(hi: pd.DataFrame, mob: pd.DataFrame) -> pd.DataFrame:
    m = mob.sort_values("mob1_at").drop_duplicates(["date_m", "addr"])
    out = hi.merge(m, left_on=["date_h", "addr"], right_on=["date_m", "addr"], how="left").drop(columns=["date_m"])
    out["mob_level"] = out.mob_level.fillna(0).astype(int)
    return out


# ---------------------------------------------------------------- 지오코딩
def adm_lookup():
    g = json.load(open(glob(str(RAW / "adm" / "*.geojson"))[0], encoding="utf-8"))
    rows = []
    for f in g["features"]:
        p = f["properties"]
        geom = shape(f["geometry"])
        c = geom.representative_point()
        toks = p["adm_nm"].split()
        rows.append(dict(sido=SIDO_ABBR.get(p["sidonm"], p["sidonm"]), sggnm=p["sggnm"].replace(" ", ""),
                         sgg_k=key_sgg(p["sggnm"]), emd=toks[-1], emd_k=key_emd(toks[-1]),
                         sgg_code=p["sgg"], lon=c.x, lat=c.y, area=geom.area))
    a = pd.DataFrame(rows)
    # 시군구 대표점 = 면적가중 평균
    s = (a.assign(wx=a.lon * a.area, wy=a.lat * a.area)
         .groupby(["sido", "sgg_k"]).agg(wx=("wx", "sum"), wy=("wy", "sum"), w=("area", "sum"), sgg_code=("sgg_code", "first"))
         .reset_index())
    s["lon"], s["lat"] = s.wx / s.w, s.wy / s.w
    return a, s[["sido", "sgg_k", "sgg_code", "lon", "lat"]]


def geocode(df: pd.DataFrame, a: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """(시도, 시군구 어간, 읍면동 어간) → 행정동 대표점, 실패 시 시군구 대표점."""
    emd_map = a.drop_duplicates(["sido", "sgg_k", "emd_k"]).set_index(["sido", "sgg_k", "emd_k"])
    emd_map2 = a.groupby(["sido", "emd_k"]).filter(lambda x: len(x) == 1).set_index(["sido", "emd_k"])  # 시도 내 유일 읍면
    sgg_map = s.set_index(["sido", "sgg_k"])
    sido_map = s.groupby("sido")[["lon", "lat"]].mean()
    out = []
    for r in df.itertuples():
        k3 = (r.g_sido, r.g_sgg, r.g_emd)
        if r.g_emd and k3 in emd_map.index:
            x = emd_map.loc[k3]
            out.append((x.lon, x.lat, x.sgg_code, "읍면동"))
        elif r.g_emd and (r.g_sido, r.g_emd) in emd_map2.index:
            x = emd_map2.loc[(r.g_sido, r.g_emd)]
            out.append((x.lon, x.lat, x.sgg_code, "읍면동"))
        elif (r.g_sido, r.g_sgg) in sgg_map.index:
            x = sgg_map.loc[(r.g_sido, r.g_sgg)]
            out.append((x.lon, x.lat, x.sgg_code, "시군구"))
        elif r.g_sido in sido_map.index:
            x = sido_map.loc[r.g_sido]
            out.append((x.lon, x.lat, None, "시도"))
        else:
            out.append((np.nan, np.nan, None, "실패"))
    df[["lon", "lat", "sgg_code", "geo_level"]] = pd.DataFrame(out, index=df.index)
    return df


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(d))


def nearest_km(df, pts):
    d = haversine(df.lon.values[:, None], df.lat.values[:, None], pts.lon.values[None, :], pts.lat.values[None, :])
    dd = np.where(np.isnan(d), np.inf, d)
    return np.where(np.isinf(dd.min(axis=1)), np.nan, dd.min(axis=1)), dd.argmin(axis=1)


# ---------------------------------------------------------------- 기상 (ASOS 일자료)
def load_asos():
    cols = {0: "tm", 1: "stn", 2: "ws_avg", 5: "ws_max", 8: "ws_ins", 10: "ta_avg", 11: "ta_max",
            18: "hm_avg", 19: "hm_min", 38: "rn_day"}
    frames = []
    for f in sorted(glob(str(RAW / "kma" / "sfcdd_*.txt"))):
        rows = [ln.split() for ln in open(f, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
        d = pd.DataFrame([[r[i] for i in cols] for r in rows if len(r) > 38], columns=list(cols.values()))
        frames.append(d)
    w = pd.concat(frames, ignore_index=True)
    w["date"] = pd.to_datetime(w.tm, format="%Y%m%d")
    w["stn"] = w.stn.astype(int)
    for c in ["ws_avg", "ws_max", "ws_ins", "ta_avg", "ta_max", "hm_avg", "hm_min", "rn_day"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
        if c != "rn_day":
            w.loc[w[c] < -8, c] = np.nan
    w.loc[w.rn_day < 0, "rn_day"] = 0.0  # -9: 무강수
    w = w.drop(columns="tm").sort_values(["stn", "date"])
    # 실효습도(He) = (1-r) Σ r^n H_(t-n), r=0.7, n=0..4  (기상청·국립산림과학원 정의)
    def eff(g):
        h = g.hm_avg
        he = sum((0.7 ** n) * h.shift(n) for n in range(5)) * 0.3
        he_prev = sum((0.7 ** n) * h.shift(n + 1) for n in range(5)) * 0.3
        wet = (g.rn_day >= 1.0).astype(int)
        # 전일까지 연속 무강수일수
        grp = wet.cumsum()
        dry = g.groupby(grp).cumcount().where(wet == 0, 0)
        return pd.DataFrame({"eff_hum": he, "eff_hum_prev": he_prev, "dry_days": dry.shift(1)}, index=g.index)
    w[["eff_hum", "eff_hum_prev", "dry_days"]] = w.groupby("stn", group_keys=False)[["hm_avg", "rn_day"]].apply(eff)
    stn = pd.read_csv(glob(str(RAW / "datagokr" / "15150429_*.csv"))[0], encoding="cp949")
    stn = stn[stn["기상대종류"].str.strip() == "ASOS"].copy()
    stn["stn"] = stn["기상대지점코드"].str.extract(r"(\d+)").astype(int)
    stn = stn.rename(columns={"경도": "lon", "위도": "lat", "기상대지점명": "stn_name"})[["stn", "stn_name", "lon", "lat"]]
    stn = stn[stn.stn.isin(w.stn.unique())].drop_duplicates("stn").reset_index(drop=True)
    return w, stn


def attach_weather(df, w, stn):
    d = haversine(df.lon.values[:, None], df.lat.values[:, None], stn.lon.values[None, :], stn.lat.values[None, :])
    order = np.argsort(np.where(np.isnan(d), np.inf, d), axis=1)[:, :3]  # 가까운 3개 지점 (결측 시 대체)
    wi = w.set_index(["stn", "date"])
    feats = ["hm_min", "hm_avg", "ws_max", "ws_ins", "ta_max", "rn_day", "eff_hum", "eff_hum_prev", "dry_days"]
    out = []
    for i, r in enumerate(df.itertuples()):
        rec = {f: np.nan for f in feats}
        rec["wx_stn"], rec["wx_km"] = np.nan, np.nan
        if not np.isnan(r.lon):
            for j in order[i]:
                key = (stn.stn.iloc[j], r.date)
                if key in wi.index and not np.isnan(wi.loc[key, "hm_min"]):
                    rec.update(wi.loc[key, feats].to_dict())
                    rec["wx_stn"], rec["wx_km"] = stn.stn_name.iloc[j], d[i, j]
                    break
        out.append(rec)
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


# ---------------------------------------------------------------- 일출·일몰 (NOAA 근사식, 결측 보완용)
def sun_times(lat, lon, day: pd.Timestamp):
    n = day.dayofyear
    g = 2 * math.pi / 365 * (n - 1)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    ha = math.degrees(math.acos(math.cos(math.radians(90.833)) / (math.cos(math.radians(lat)) * math.cos(decl)) - math.tan(math.radians(lat)) * math.tan(decl)))
    noon = 720 - 4 * lon - eqt + 540  # KST(+9h) 분
    return pd.Timedelta(minutes=noon - 4 * ha), pd.Timedelta(minutes=noon + 4 * ha)


# ---------------------------------------------------------------- 대형산불위험예보
def attach_forecast(df, fc):
    fc = fc.assign(date=fc.t.dt.normalize(), hour=fc.t.dt.hour)
    by_sgg = fc.groupby(["sido", "sgg_k", "date"])
    first_hour = by_sgg.hour.min().rename("fc_first_hour")
    n_hours = by_sgg.t.nunique().rename("fc_hours")
    agg = pd.concat([first_hour, n_hours, by_sgg.eff_hum.min().rename("fc_min_effhum"), by_sgg.wind.max().rename("fc_max_wind")], axis=1).reset_index()
    out = df.merge(agg, left_on=["g_sido", "g_sgg", "date"], right_on=["sido", "sgg_k", "date"], how="left", suffixes=("", "_fc"))
    out = out.drop(columns=[c for c in ["sido_fc", "sgg_k_fc"] if c in out.columns])
    out["warn_day"] = out.fc_hours.notna().astype(int)  # 발생일 해당 시군구에 주의보 발령 시각이 1번이라도 존재
    out["warn_before"] = (out.fc_first_hour <= out.occ.dt.hour).fillna(False).astype(int)  # 발생 시각 이전(포함) 발령
    # 신고 시점에 이미 알 수 있는 예보 정보만: 발생 시각까지의 해당 시군구 주의보 시간 수, 전국 주의보 시군구 수, 전일 주의보
    by_date = {d: g for d, g in fc.groupby("date")}
    hb, nat, prev = [], [], []
    for r in out.itertuples():
        g = by_date.get(r.date)
        h = r.occ.hour
        if g is None:
            hb.append(0)
            nat.append(0)
        else:
            gb = g[g.hour <= h]
            hb.append(int(gb[(gb.sido == r.g_sido) & (gb.sgg_k == r.g_sgg)].hour.nunique()))
            nat.append(int(gb[["sido", "sgg_k"]].drop_duplicates().shape[0]))
        gp = by_date.get(r.date - pd.Timedelta(days=1))
        prev.append(int(gp is not None and ((gp.sido == r.g_sido) & (gp.sgg_k == r.g_sgg)).any()))
    out["warn_hours_before"] = hb
    out["nat_warn_sgg_before"] = nat
    out["warn_prevday"] = prev
    return out


# ---------------------------------------------------------------- 동시다발성
def attach_concurrency(df):
    occ = df.occ.values
    end = df.ext_best.values
    n_active, n_before_day, n_active_sido = [], [], []
    sidos = df.g_sido.values
    days = df.date.values
    for i in range(len(df)):
        act = (occ < occ[i]) & (end > occ[i])
        n_active.append(int(act.sum()))
        n_active_sido.append(int((act & (sidos == sidos[i])).sum()))
        n_before_day.append(int(((days == days[i]) & (occ < occ[i])).sum()))
    df["n_active_nat"] = n_active
    df["n_active_sido"] = n_active_sido
    df["n_before_today"] = n_before_day
    df["n_fires_today"] = df.groupby("date").fire_no.transform("size")
    return df


def main():
    st, hi, ri, mob, fc = load_stats(), load_history(), load_risk(), load_mobilization(), load_forecast()
    res = load_resources()
    hi = attach_mob(attach_risk(hi, ri), mob)
    mt = match_stats_history(st, hi)
    df = st.merge(mt, on="fire_no", how="left").merge(hi, on="fire_id", how="left")
    print(f"통계↔이력 매칭 {mt.fire_id.notna().sum()}/{len(st)}  위험지수 결측 {df.danger_index.isna().mean():.1%}")

    # 지오코딩 키: 이력의 정식 주소 우선, 없으면 통계의 약칭
    df["g_sido"] = df.sido_h.fillna(df.sido)
    df["g_sgg"] = np.where(df.sgg_k_h.fillna("") != "", df.sgg_k_h, df.sgg_k)
    df["g_emd"] = np.where(df.emd_k_h.fillna("") != "", df.emd_k_h, df.emd_k)
    # 행정구역 변경 보정: 군위군 2023.7 경북→대구 편입 / 세종은 통계의 '시군구' 칸에 읍면이 기재됨
    gunwi = df.g_sgg == "군위"
    df.loc[gunwi, "g_sido"] = "대구"
    sj = df.g_sido == "세종"
    df.loc[sj & (df.g_emd == ""), "g_emd"] = df.loc[sj & (df.g_emd == ""), "sgg_raw"].map(key_emd)
    df.loc[sj, "g_sgg"] = "세종"
    a, s = adm_lookup()
    df = geocode(df, a, s)
    print("지오코딩:", df.geo_level.value_counts().to_dict())

    # 자원 접근성
    df["heli_km"], idx = nearest_km(df, res["heli"])
    df["heli_base"] = res["heli"].대기장소명.str.strip().values[idx]
    df["crew_km"], _ = nearest_km(df, res["crew"])
    df["dam_km"], _ = nearest_km(df, res["dam"])

    # 시간 변수
    df["attack_delay_min"] = df.attack_gap_raw.where(df.attack_gap_raw.between(0, 24 * 60))
    df["ext_best"] = df.ext.fillna(df.attack_end)
    df["dur_total_min"] = (df.ext_best - df.occ).dt.total_seconds() / 60
    df.loc[df.dur_total_min < 0, "dur_total_min"] = np.nan
    # 이력 데이터의 일출·일몰은 같은 날이면 전국이 같은 값(지점 무관) → 위치별로 다시 계산하고 원값은 품질 점검용으로 보존
    df["sunset_kfs"] = df.sunset
    sun = [sun_times(la, lo, d) for la, lo, d in zip(df.lat, df.lon, df.date)]
    df["sunrise"] = [s[0] for s in sun]
    df["sunset"] = [s[1] for s in sun]
    tod = df.occ - df.date
    df["min_to_sunset"] = (df.sunset - tod).dt.total_seconds() / 60
    df["is_night"] = ((tod < df.sunrise) | (tod > df.sunset)).astype(int)
    df["contained_before_sunset"] = (df.ext_best <= df.date + df.sunset).astype(int)
    df["hour"] = df.occ.dt.hour
    df["month"] = df.occ.dt.month
    df["year"] = df.occ.dt.year
    df["weekday"] = df.occ.dt.weekday
    kr = holidays.KR(years=range(2021, 2027))
    df["is_holiday"] = df.date.dt.date.map(lambda d: d in kr).astype(int)
    df["is_offday"] = ((df.weekday >= 5) | (df.is_holiday == 1)).astype(int)

    # 기상
    w, stn = load_asos()
    df = attach_weather(df, w, stn)
    # 예보·동시성
    df = attach_forecast(df, fc)
    df = attach_concurrency(df)

    # 목표변수
    # 산림청 대형산불 정의는 '100ha 이상 또는 24시간 이상'이지만, 통계의 진화종료시각과 이력의 진화완료시각이
    # 크게 어긋나는 건이 많아(품질 점검 참조) 지속시간 기준은 쓰지 않고 면적 기준만 사용
    df["y_1ha"] = (df.area_ha >= 1).astype(int)
    df["y_10ha"] = (df.area_ha >= 10).astype(int)
    df["y_100ha"] = (df.area_ha >= 100).astype(int)

    keep = ["fire_no", "fire_id", "occ", "date", "year", "month", "hour", "weekday", "is_holiday", "is_offday",
            "office", "g_sido", "sgg_raw", "emd_raw", "addr", "agency", "sgg_code", "lon", "lat", "geo_level",
            "cause", "cause_code", "cause_detail", "cause_etc", "area_ha", "y_1ha", "y_10ha", "y_100ha",
            "ext_best", "dur_total_min", "attack_start", "attack_end", "attack_delay_min", "data_error_flag",
            "sunrise", "sunset", "sunset_kfs", "min_to_sunset", "is_night", "contained_before_sunset",
            "report_type", "danger_index", "sms_request", "mob_level", "mob1_at", "mob2_at", "mob3_at",
            "heli_km", "heli_base", "crew_km", "dam_km",
            "wx_stn", "wx_km", "hm_min", "hm_avg", "ws_max", "ws_ins", "ta_max", "rn_day", "eff_hum", "eff_hum_prev", "dry_days",
            "warn_day", "warn_before", "warn_hours_before", "nat_warn_sgg_before", "warn_prevday",
            "fc_first_hour", "fc_hours", "fc_min_effhum", "fc_max_wind",
            "n_active_nat", "n_active_sido", "n_before_today", "n_fires_today", "match_score"]
    df = df[keep]
    for c in ["sunrise", "sunset", "sunset_kfs"]:
        df[c] = df[c].astype(str).str.replace("0 days ", "", regex=False)
    df.to_csv(PROC / "fires.csv", index=False, encoding="utf-8-sig")
    w.to_csv(PROC / "asos_daily.csv.gz", index=False)
    stn.to_csv(PROC / "asos_stations.csv", index=False, encoding="utf-8-sig")
    print("저장:", PROC / "fires.csv", df.shape)
    print(df[["y_1ha", "y_10ha", "y_100ha"]].sum().to_dict())
    print(df[["attack_delay_min", "min_to_sunset", "danger_index", "eff_hum", "ws_max", "heli_km", "n_active_nat"]].describe().T.round(2))


if __name__ == "__main__":
    main()
