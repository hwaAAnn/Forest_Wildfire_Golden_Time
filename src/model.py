"""신고 시점 '대형화(10ha 이상) 위험' 트리아지 모델.

- 입력: 신고 시점에 이미 알 수 있는 정보만 (발생원인·진화시간·동원령 같은 사후 정보 제외)
  · 기상은 당일 일자료를 '예보값의 대용'으로 사용 (운영 시 기상청 단기예보로 대체)
  · 산불위험지수 결측(37.7%)은 기록 관행에 따른 결측(대형 산불일수록 기록률↑)이므로
    결측 자체가 정답을 흘리지 않도록 '당일 전국 중앙값'으로 대체
- 검증: ① 연도 교차검증(LOYO: 한 해를 통째로 빼고 학습 → 그 해 예측)
        ② 시간 외 검증(2022~2024 학습 → 2025 예측)
- 비교: 단일 지표 vs 로지스틱 회귀 vs LightGBM vs 앙상블, 데이터 결합 단계별(ablation)
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PROC

TARGET = "y_10ha"
FEATURE_GROUPS = {
    "산림청: 위험지수·대형산불위험예보": ["danger_index", "warn_before", "warn_hours_before", "nat_warn_sgg_before", "warn_prevday"],
    "산림청: 동시다발(상황관제)": ["n_active_nat", "n_active_sido", "n_before_today"],
    "산림청: 진화자원 접근성": ["heli_km", "crew_km", "dam_km"],
    "시간·달력": ["month", "doy_sin", "doy_cos", "hour_f", "is_offday", "min_to_sunset"],
    "기상청: 기상": ["eff_hum", "eff_hum_prev", "hm_min", "ws_max", "ws_ins", "ta_max", "dry_days"],
    "위치": ["lat", "lon"],
}
SETS = {  # 데이터 결합 단계별 비교
    "① 산림청 데이터만": ["산림청: 위험지수·대형산불위험예보", "산림청: 동시다발(상황관제)", "산림청: 진화자원 접근성", "시간·달력"],
    "② + 기상청 기상": ["산림청: 위험지수·대형산불위험예보", "산림청: 동시다발(상황관제)", "산림청: 진화자원 접근성", "시간·달력", "기상청: 기상"],
    "③ + 위치(최종)": list(FEATURE_GROUPS),
}
LABEL = {
    "month": "월", "doy_sin": "연중일(sin)", "doy_cos": "연중일(cos)", "hour_f": "발생 시각", "is_offday": "주말·공휴일",
    "min_to_sunset": "일몰까지 남은 시간", "eff_hum": "실효습도(당일)", "eff_hum_prev": "실효습도(전일)", "hm_min": "최소습도",
    "ws_max": "최대풍속", "ws_ins": "최대순간풍속", "ta_max": "최고기온", "dry_days": "연속 무강수일",
    "danger_index": "산불위험지수", "warn_before": "대형산불주의보(해당 시군구)", "warn_hours_before": "주의보 누적 시간",
    "nat_warn_sgg_before": "전국 주의보 발령 시군구 수", "warn_prevday": "전일 주의보",
    "n_active_nat": "전국 동시 진행 산불 수", "n_active_sido": "시도 내 동시 진행 산불 수", "n_before_today": "당일 앞선 산불 수",
    "lat": "위도", "lon": "경도", "heli_km": "최근접 산림항공관리소 거리", "crew_km": "최근접 진화대 대기장소 거리",
    "dam_km": "최근접 담수용 사방댐 거리",
}
MONOTONE = {"eff_hum": -1, "eff_hum_prev": -1, "hm_min": -1, "ws_max": 1, "ws_ins": 1, "dry_days": 1,
            "danger_index": 1, "warn_before": 1, "warn_hours_before": 1, "nat_warn_sgg_before": 1,
            "n_active_nat": 1, "n_active_sido": 1, "n_before_today": 1}
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, n_estimators=350, num_leaves=7, min_child_samples=25,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=4)


def feats(set_name):
    return [f for g in SETS[set_name] for f in FEATURE_GROUPS[g]]


def load():
    d = pd.read_csv(PROC / "fires.csv", parse_dates=["occ", "date", "mob1_at", "mob2_at", "mob3_at"])
    doy = d.occ.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["hour_f"] = d.occ.dt.hour + d.occ.dt.minute / 60
    d["danger_index_raw"] = d.danger_index
    day_med = d.groupby("date").danger_index.transform("median")
    mon_med = d.groupby("month").danger_index.transform("median")
    d["danger_index"] = d.danger_index.fillna(day_med).fillna(mon_med)
    return d


def fit_predict(kind, Xtr, ytr, Xte, cols, seeds=(0, 1, 2, 3, 4)):
    if kind == "lr":
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1], [m]
    mono = [MONOTONE.get(f, 0) for f in cols]
    preds, models = [], []
    for s in seeds:
        m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=s, monotone_constraints=mono)
        m.fit(Xtr, ytr)
        preds.append(m.predict_proba(Xte)[:, 1])
        models.append(m)
    return np.mean(preds, axis=0), models


def rank01(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def loyo(d, cols):
    """연도 교차검증 OOF 예측: lr, lgb, ens(순위 평균)."""
    out = {k: np.zeros(len(d)) for k in ("lr", "lgb")}
    for yr in sorted(d.year.unique()):
        tr, te = (d.year != yr).values, (d.year == yr).values
        for k in out:
            out[k][te], _ = fit_predict(k, d.loc[tr, cols], d.loc[tr, TARGET], d.loc[te, cols], cols)
    out["ens"] = 0.5 * rank01(out["lr"]) + 0.5 * rank01(out["lgb"])
    return out


def capture(d, score, frac=0.10, n_rep=200, seed=0):
    """점수 상위 frac만 '우선 대응'할 때 포착률(동점은 무작위 순서의 기댓값)."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(frac * len(d)))
    y10, y100, area = d.y_10ha.values, d.y_100ha.values, d.area_ha.values
    acc = np.zeros(4)
    for _ in range(n_rep):
        order = np.lexsort((rng.random(len(d)), -np.asarray(score)))[:k]
        acc += [y10[order].sum() / y10.sum(), y100[order].sum() / max(y100.sum(), 1),
                area[order].sum() / area.sum(), y10[order].mean()]
    acc /= n_rep
    return dict(zip(["cap_10ha", "cap_100ha", "cap_area", "precision"], acc))


def evaluate(d, score, name):
    y, s = d[TARGET].values, np.asarray(score, dtype=float)
    r = dict(model=name, auc=roc_auc_score(y, s), ap=average_precision_score(y, s), base_rate=y.mean())
    r.update({f"{k}@10%": v for k, v in capture(d, s).items()})
    return r


def single_indicators(d):
    return {
        "산불위험지수(산림청)": d.danger_index,
        "대형산불주의보(국립산림과학원)": d.warn_before + d.warn_hours_before / 100,
        "실효습도(기상청, 낮을수록 위험)": -d.eff_hum.fillna(d.eff_hum.median()),
        "최대풍속(기상청)": d.ws_max.fillna(d.ws_max.median()),
        "전국 동시 진행 산불 수(산림청)": d.n_active_nat.astype(float),
    }


def main():
    d = load()
    final = "③ + 위치(최종)"
    cols = feats(final)

    # ① 연도 교차검증 + 데이터 결합 단계별 비교
    rows = [evaluate(d, s, n) for n, s in single_indicators(d).items()]
    oof = {}
    for set_name in SETS:
        oof[set_name] = loyo(d, feats(set_name))
        for k, nm in (("lr", "로지스틱"), ("lgb", "LightGBM"), ("ens", "앙상블")):
            rows.append(evaluate(d, oof[set_name][k], f"{set_name} · {nm}"))
    loyo_tbl = pd.DataFrame(rows)
    d["score"] = oof[final]["ens"]
    d["score_lgb"], d["score_lr"] = oof[final]["lgb"], oof[final]["lr"]

    # ② 시간 외 검증: 2022~2024 → 2025
    tr, te = (d.year <= 2024).values, (d.year == 2025).values
    p_lr, _ = fit_predict("lr", d.loc[tr, cols], d.loc[tr, TARGET], d.loc[te, cols], cols)
    p_gb, _ = fit_predict("lgb", d.loc[tr, cols], d.loc[tr, TARGET], d.loc[te, cols], cols)
    d25 = d[te].copy()
    fwd = [evaluate(d25, s[te], n) for n, s in single_indicators(d).items()]
    fwd += [evaluate(d25, p_lr, "로지스틱"), evaluate(d25, p_gb, "LightGBM"),
            evaluate(d25, 0.5 * rank01(p_lr) + 0.5 * rank01(p_gb), "앙상블")]
    fwd_tbl = pd.DataFrame(fwd)

    # ③ 전체 데이터 LightGBM → SHAP
    import shap
    _, models = fit_predict("lgb", d[cols], d[TARGET], d[cols], cols)
    sv = np.mean([np.asarray(shap.TreeExplainer(m).shap_values(d[cols])) for m in models], axis=0)
    if sv.ndim == 3:
        sv = sv[1]
    grp = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    imp = pd.DataFrame({"feature": cols, "label": [LABEL[f] for f in cols], "group": [grp[f] for f in cols],
                        "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    np.save(PROC / "shap_values.npy", sv)
    pd.Series(cols).to_csv(PROC / "shap_columns.csv", index=False)

    # ④ 동원령 대비 선행시간 (OOF 앙상블 점수 상위 10% 기준선)
    thr = np.quantile(d.score, 0.90)
    m = d[d[["mob1_at", "mob2_at", "mob3_at"]].notna().any(axis=1)].copy()
    m["first_mob"] = m[["mob1_at", "mob2_at", "mob3_at"]].min(axis=1)
    m["lead_min"] = (m.first_mob - m.occ).dt.total_seconds() / 60
    m["flagged"] = m.score >= thr
    lead = dict(n_mobilized=int(len(m)), n_flagged=int(m.flagged.sum()),
                n_mob3=int(m.mob3_at.notna().sum()), n_mob3_flagged=int((m.flagged & m.mob3_at.notna()).sum()),
                median_lead_min_flagged=float(m.loc[m.flagged, "lead_min"].median()),
                median_lead3_min_flagged=float(((m.mob3_at - m.occ).dt.total_seconds() / 60)[m.flagged].median()),
                flagged_area_share=float(m.loc[m.flagged, "area_ha"].sum() / m.area_ha.sum()))

    # ⑤ 동시다발일(하루 10건+) 우선 대응 시뮬레이션: 같은 슬롯 수를 선착순 vs 점수(실시간 임계값)로 배정
    sim = []
    for day, g in d[d.n_fires_today >= 10].groupby("date"):
        g = g.sort_values("occ")
        tri = (g.score >= thr).values
        k = int(tri.sum())
        fifo = np.zeros(len(g), bool)
        fifo[:k] = True
        sim.append(dict(date=day.date().isoformat(), n=len(g), slots=k, y10=int(g.y_10ha.sum()),
                        fifo_hit=int(g.y_10ha.values[fifo].sum()), triage_hit=int(g.y_10ha.values[tri].sum()),
                        fifo_area=float(g.area_ha.values[fifo].sum()), triage_area=float(g.area_ha.values[tri].sum()),
                        area=float(g.area_ha.sum())))
    sim = pd.DataFrame(sim)

    pd.set_option("display.width", 250)
    print("\n[연도 교차검증 LOYO]\n", loyo_tbl.round(3).to_string(index=False))
    print("\n[시간 외 검증 2022~24 → 2025]\n", fwd_tbl.round(3).to_string(index=False))
    print("\n[SHAP 상위]\n", imp.head(12).round(4).to_string(index=False))
    print("\n[동원령 대비 선행시간]", lead)
    print("\n[동시다발일 시뮬레이션]", sim[["n", "slots", "y10", "fifo_hit", "triage_hit"]].sum().to_dict(),
          f"면적 선착순 {sim.fifo_area.sum() / sim.area.sum():.1%} vs 트리아지 {sim.triage_area.sum() / sim.area.sum():.1%}")

    d.loc[te, "score_fwd"] = 0.5 * rank01(p_lr) + 0.5 * rank01(p_gb)
    keep = ["fire_no", "occ", "date", "year", "g_sido", "sgg_raw", "emd_raw", "addr", "lon", "lat", "area_ha",
            "y_1ha", "y_10ha", "y_100ha", "score", "score_lgb", "score_lr", "score_fwd", "danger_index",
            "danger_index_raw", "warn_before", "nat_warn_sgg_before", "n_active_nat", "eff_hum", "ws_max", "ws_ins",
            "mob_level", "n_fires_today"]
    d[keep].to_csv(PROC / "predictions.csv", index=False, encoding="utf-8-sig")
    loyo_tbl.to_csv(PROC / "metrics_loyo.csv", index=False, encoding="utf-8-sig")
    fwd_tbl.to_csv(PROC / "metrics_forward2025.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(PROC / "shap_importance.csv", index=False, encoding="utf-8-sig")
    sim.to_csv(PROC / "triage_simulation.csv", index=False, encoding="utf-8-sig")
    m[["occ", "addr", "area_ha", "mob_level", "lead_min", "score", "flagged"]].to_csv(
        PROC / "mobilization_lead.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(threshold_top10=float(thr), lead=lead), open(PROC / "model_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
