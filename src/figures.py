"""제출용 그림 (figures/*.png). 각 그림은 제목이 곧 결론이 되도록 구성.

색: 기본 검증 팔레트(주황=산불·강조, 파랑=결합 모델, 회색=맥락), 얇은 막대, 실선 헤어라인 그리드.
"""
import json
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.stats import rankdata

from config import FIG, PROC, RAW

SURF, PLANE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAYMARK = "#c3c2b7"
LAND = "#f0efec"

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titlepad": 10, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
})
DPI = 200
SRC = "자료: 산림청·국립산림과학원 공공데이터(공공데이터포털), 기상청 API허브 ASOS, 통계청 SGIS 행정동 경계(vuski/admdongkor)"


def fig_base(w=8, h=4.5):
    return plt.figure(figsize=(w, h), dpi=DPI)


def headline(fig, title, sub=None, y=0.955):
    fig.text(0.035, y, title, fontsize=15, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.035, y - 0.075, sub, fontsize=9.5, color=INK2, va="top", linespacing=1.5)


def footer(fig, text=SRC):
    fig.text(0.035, 0.022, text, fontsize=7, color=MUTED, va="bottom")


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(grid != "y")
    ax.spines["bottom"].set_color(BASE)
    ax.spines["left"].set_color(BASE)
    ax.tick_params(length=0)
    if grid:
        ax.grid(True, axis=grid)
        ax.set_axisbelow(True)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def rate_bars(ax, d, col, bins, labels, title, color=ORANGE, base=None, xlabel=None):
    g = pd.cut(d[col], bins, labels=labels, right=False)
    t = d.groupby(g, observed=False).agg(n=("y_10ha", "size"), k=("y_10ha", "sum"))
    t["p"] = t.k / t.n
    x = np.arange(len(t))
    ax.bar(x, 100 * t.p, width=0.5, color=color, zorder=2)
    for i, (k, n, p) in enumerate(zip(t.k, t.n, t.p)):
        lo, hi = wilson(k, n)
        ax.plot([i, i], [100 * lo, 100 * hi], color=INK2, lw=0.8, zorder=3)
        ax.text(i + 0.07, 100 * p + 0.25, f"{100 * p:.1f}%", ha="left", va="bottom", fontsize=8.2, color=INK, zorder=4)
    ax.set_xticks(x, [f"{lb}\n(n={n:,})" for lb, n in zip(labels, t.n)], fontsize=7.5)
    if base is not None:
        ax.axhline(100 * base, color=MUTED, lw=0.8, zorder=1)
    ax.set_title(title, fontsize=10.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    clean(ax)
    return t


SIDO_SHORT = {"서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천", "광주광역시": "광주",
              "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기", "강원특별자치도": "강원",
              "충청북도": "충북", "충청남도": "충남", "전북특별자치도": "전북", "전라남도": "전남", "경상북도": "경북",
              "경상남도": "경남", "제주특별자치도": "제주"}


def place(r, with_date=True):
    """이력 데이터의 정식 주소(예: '경상북도 의성군 안평면') → '2025.3 경북 의성군 안평면'."""
    if isinstance(r.addr, str):
        p = r.addr.split()
        nm = " ".join([SIDO_SHORT.get(p[0], p[0])] + p[1:3])
    else:
        nm = f"{r.g_sido} {r.sgg_raw if isinstance(r.sgg_raw, str) else ''} {r.emd_raw if isinstance(r.emd_raw, str) else ''}".strip()
    return f"{r.occ.year}.{r.occ.month} {nm}" if with_date else nm


# ------------------------------------------------------------------ 0. 데이터 결합 구조
def fig0_pipeline(d):
    fig = fig_base(8, 4.6)
    headline(fig, "산림청 공공데이터 7종 + 기상·행정경계를 '산불 1건 = 1행'으로 결합",
             f"공통 ID가 없는 데이터들을 지명 정규화·시각 근접 매칭·지오코딩으로 연결 → 2,020건 × 70여 개 변수 통합 테이블")
    ax = fig.add_axes([0.02, 0.07, 0.96, 0.72])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge=BASE, fill=SURF, tcolor=INK, body_dy=5.2, bsize=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.4", fc=fill, ec=edge, lw=0.9))
        ax.text(x + 1.2, y + h - 1.6, title, fontsize=8.2, fontweight="bold", color=tcolor, va="top")
        if body:
            ax.text(x + 1.2, y + h - body_dy, body, fontsize=bsize, color=INK2, va="top", linespacing=1.5)

    kfs = [("산불통계데이터", "2,020건: 발생·진화종료 시각, 장소, 원인, 면적"),
           ("산불상태별 이력", "진화 시작·완료 시각, 일출·일몰"),
           ("산불신고 유형별 위험지수", "신고 시점 산불위험지수"),
           ("산불 단계별 동원력 이력", "동원령 1~3단계 발령 시각(60건)"),
           ("대형산불위험예보목록 (국립산림과학원)", "읍면동·시간별 주의보 49,117건"),
           ("진화대원 대기장소 · 담수용 사방댐", "산림항공관리소 등 진화자원 위경도")]
    ax.text(1, 100, "산림청 공공데이터 (파일 7종)", fontsize=8.5, color=INK, fontweight="bold", va="top")
    for i, (t, b) in enumerate(kfs):
        box(1, 81.5 - i * 14.0, 33, 10.6, t, b, edge=ORANGE)
    ax.text(1, -1, "+ 연계 데이터: 기상청 ASOS 일자료(API허브) · ASOS 지점 좌표(국립환경과학원) · 통계청 행정동 경계 · 공휴일",
            fontsize=7.0, color=INK2, va="bottom")

    steps = ["① 지명 정규화 + 시각 근접 매칭 (98.9% 결합)",
             "② 행정동 대표점으로 지오코딩",
             "③ 최근접 ASOS → 실효습도·풍속·무강수일",
             "④ 신고 시각의 전국 동시 진행 산불 수",
             "⑤ 신고 전 대형산불주의보·전국 발령 수",
             "⑥ 산림항공관리소·대기장소·사방댐 거리",
             "⑦ 위치별 일몰 재계산 · 원인 표준화"]
    box(39, 12, 30, 80, "결합 · 정제", "\n".join(steps), edge=INK2, body_dy=8, bsize=7.6)
    for i in range(6):
        y = 81.5 - i * 14.0 + 5.3
        ax.annotate("", xy=(38.7, 52), xytext=(34.4, y), arrowprops=dict(arrowstyle="-", color=BASE, lw=0.7))
    ax.annotate("", xy=(73.7, 52), xytext=(69.4, 52), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0))
    box(74, 60, 25, 32, "통합 분석 테이블", f"산불 {len(d):,}건 × 70여 개 변수\n2022.1 ~ 2025.9\n\n결과: 피해면적, 10ha·100ha 이상\n모델 입력: 신고 시점 정보만", edge=BLUE, body_dy=6.5)
    box(74, 12, 25, 42, "활용", "① 피해 집중도·동시다발 분석\n② 위험 신호별 대형화 확률\n③ 신고 즉시 대형화 위험\n    트리아지 모델\n④ 데이터 품질 점검 17건\n    → 개선 제안", edge=BLUE, body_dy=6.5)
    ax.annotate("", xy=(86.5, 54.8), xytext=(86.5, 59.4), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0))
    footer(fig)
    fig.savefig(FIG / "00_pipeline.png")
    plt.close(fig)


# ------------------------------------------------------------------ 1. 피해 집중도
def fig1_concentration(d):
    fig = fig_base(8, 4.6)
    s = d.sort_values("area_ha", ascending=False).reset_index(drop=True)
    tot = s.area_ha.sum()
    top20 = s.area_ha[:20].sum() / tot
    headline(fig, f"산불 2,020건 중 상위 1%(20건)가 피해면적의 {100 * top20:.0f}%를 태웠다",
             f"2022.1~2025.9 전체 피해면적 {tot:,.0f}ha · 산불 1건의 피해면적 중앙값 {s.area_ha.median():.2f}ha")
    # 왼쪽: 핵심 수치
    tiles = [("상위 1% (20건)", f"{100 * top20:.1f}%", "전체 피해면적 중 비중"),
             ("상위 2건 (2025.3 의성)", f"{100 * s.area_ha[:2].sum() / tot:.0f}%", "의성 안계·안평 두 발화점"),
             ("나머지 2,000건", f"{100 * (1 - top20):.1f}%", f"합계 {s.area_ha[20:].sum():,.0f}ha")]
    for i, (lab, val, sub) in enumerate(tiles):
        y = 0.70 - i * 0.215
        fig.text(0.04, y, lab, fontsize=8.5, color=INK2)
        fig.text(0.04, y - 0.085, val, fontsize=21, color=INK, fontweight="bold")
        fig.text(0.04, y - 0.125, sub, fontsize=7.5, color=MUTED)
    ax = fig.add_axes([0.44, 0.14, 0.52, 0.62])
    rows = [(place(r), r.area_ha, ORANGE) for r in s.head(10).itertuples()]
    rows.append(("11~20위 (10건) 합계", s.area_ha[10:20].sum(), GRAYMARK))
    rows.append(("나머지 2,000건 합계", s.area_ha[20:].sum(), GRAYMARK))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.55, color=[r[2] for r in rows], zorder=2)
    for yi, (lab, v, _) in zip(y, rows):
        ax.text(v + 600, yi, f"{v:,.0f}ha", va="center", fontsize=7.8, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.8)
    ax.set_xticks(range(0, 60001, 10000), ["0"] + [f"{k}만" for k in range(1, 7)], fontsize=8)
    ax.set_xlim(0, 62000)
    ax.set_xlabel("피해면적(ha)", fontsize=8, color=MUTED)
    clean(ax, grid="x")
    ax.spines["left"].set_visible(False)
    footer(fig, "자료: 산림청_산불통계데이터(공공데이터포털, 2025.11.10 기준 파일) · 상위 10건은 개별, 이후는 합계")
    fig.savefig(FIG / "01_concentration.png")
    plt.close(fig)


# ------------------------------------------------------------------ 2. 동시다발
def fig2_concurrency(d):
    fig = fig_base(8, 4.6)
    base = d.y_10ha.mean()
    t = d.groupby(pd.cut(d.n_active_nat, [0, 1, 2, 4, 7, 11, 999], right=False), observed=False).y_10ha.mean()
    ratio = t.iloc[-1] / t.iloc[0]
    big = d[d.n_fires_today >= 16]
    headline(fig, f"동시에 타는 산불이 많을수록 대형화 확률이 최대 {ratio:.0f}배로 커진다",
             f"신고 시점에 전국에서 다른 산불이 11건 이상 타고 있으면 10ha 이상으로 커질 확률 {100 * t.iloc[-1]:.1f}% (없을 때 {100 * t.iloc[0]:.1f}%)\n"
             f"하루 16건 이상 동시다발한 {big.date.nunique()}일에 전체 산불의 {100 * len(big) / len(d):.1f}%, 피해면적의 {100 * big.area_ha.sum() / d.area_ha.sum():.0f}%가 몰렸다")
    ax = fig.add_axes([0.07, 0.17, 0.5, 0.5])
    rate_bars(ax, d, "n_active_nat", [0, 1, 2, 4, 7, 11, 999], ["0건", "1건", "2~3건", "4~6건", "7~10건", "11건+"],
              "신고 시점 전국 동시 진행 산불 수별 10ha 이상 확률 (가로선: 전체 평균)", base=base)
    ax.set_ylim(0, 30)
    # 오른쪽: 건수 vs 면적 비중 (분절 막대)
    ax2 = fig.add_axes([0.63, 0.24, 0.33, 0.36])
    share_n = len(big) / len(d)
    share_a = big.area_ha.sum() / d.area_ha.sum()
    for i, (lab, sh) in enumerate([("산불 건수", share_n), ("피해면적", share_a)]):
        yy = 1 - i
        ax2.barh(yy, sh, height=0.42, color=ORANGE, zorder=2)
        ax2.barh(yy, 1 - sh - 0.006, left=sh + 0.006, height=0.42, color=GRAYMARK, zorder=2)
        if sh > 0.12:
            ax2.text(0.02, yy, f"{100 * sh:.1f}%", va="center", color="white", fontsize=9, fontweight="bold")
        else:
            ax2.text(sh + 0.02, yy, f"{100 * sh:.1f}%", va="center", color=INK, fontsize=9, fontweight="bold")
    ax2.set_yticks([1, 0], ["산불 건수", "피해면적"], fontsize=8.5)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_title(f"동시다발 {big.date.nunique()}일(하루 16건+)의 비중", fontsize=10.5)
    ax2.scatter([], [], marker="s", color=ORANGE, label=f"동시다발 {big.date.nunique()}일")
    ax2.scatter([], [], marker="s", color=GRAYMARK, label=f"나머지 {d.date.nunique() - big.date.nunique()}일")
    ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, frameon=False, fontsize=8, handletextpad=0.3)
    days = ", ".join(pd.to_datetime(sorted(big.date.unique())).strftime("%y.%m.%d"))
    fig.text(0.63, 0.115, f"해당일: {days}", fontsize=7, color=MUTED)
    footer(fig, "동시 진행 = 신고 시각에 이미 발생했고 아직 진화종료 전인 다른 산불 · 막대 위 선: 95% 신뢰구간 · 자료: 산림청_산불통계데이터·산불상태별 이력")
    fig.savefig(FIG / "02_concurrency.png")
    plt.close(fig)


# ------------------------------------------------------------------ 3. 위험 신호
def fig3_signals(d):
    fig = fig_base(8, 5.4)
    base = d.y_10ha.mean()
    headline(fig, "건조·강풍·높은 위험지수·주의보 — 대형화 확률을 평균의 3~5배로",
             f"조건별로 신고된 산불이 10ha 이상으로 커진 비율 · 세로선: 95% 신뢰구간 · 가로선: 전체 평균 {100 * base:.1f}%", y=0.965)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.76, bottom=0.12, hspace=0.9, wspace=0.18)
    rate_bars(fig.add_subplot(gs[0, 0]), d, "eff_hum", [0, 30, 35, 40, 45, 50, 101],
              ["<30", "30~35", "35~40", "40~45", "45~50", "50+"], "실효습도(%) — 기상청 ASOS로 계산", base=base)
    rate_bars(fig.add_subplot(gs[0, 1]), d, "ws_ins", [0, 6, 8, 10, 12, 14, 99],
              ["<6", "6~8", "8~10", "10~12", "12~14", "14+"], "최대순간풍속(m/s) — 기상청 ASOS", base=base)
    di = d[d.danger_index.notna()]
    rate_bars(fig.add_subplot(gs[1, 0]), di, "danger_index", [0, 30, 45, 60, 70, 80, 101],
              ["<30", "30~45", "45~60", "60~70", "70~80", "80+"], "산불위험지수(신고 시점) — 산림청", base=base)
    ax = fig.add_subplot(gs[1, 1])
    rate_bars(ax, d, "warn_day", [0, 1, 2], ["미발령", "발령"], "대형산불주의보(같은 날·시군구) — 국립산림과학원", base=base)
    warned = d[d.warn_day == 1]
    ax.text(-0.4, 28, f"주의보 발령 시군구 산불\n{len(warned)}건({100 * len(warned) / len(d):.1f}%)이 전체\n피해면적의 {100 * warned.area_ha.sum() / d.area_ha.sum():.0f}% 차지",
            fontsize=7.8, color=INK, va="top", linespacing=1.4)
    for a in fig.axes:
        a.set_ylim(0, 30)
    footer(fig, "산불위험지수는 기록된 1,26x건 기준(결측 36.9%) · 자료: 산림청·국립산림과학원 공공데이터, 기상청 API허브".replace("1,26x", f"{len(di):,}"))
    fig.savefig(FIG / "03_signals.png")
    plt.close(fig)


# ------------------------------------------------------------------ 4. 2025 시간 외 검증
def curves(df, score, seed=0):
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(df)), -np.asarray(score)))
    y10 = df.y_10ha.values[order].cumsum() / df.y_10ha.sum()
    ar = df.area_ha.values[order].cumsum() / df.area_ha.sum()
    x = np.arange(1, len(df) + 1) / len(df)
    return np.r_[0, x], np.r_[0, y10], np.r_[0, ar]


def fig4_forward(p, fires):
    p25 = p[p.year == 2025].copy()
    f25 = fires[fires.year == 2025].set_index("fire_no").loc[p25.fire_no]
    p25["di"] = p25.danger_index.values
    fig = fig_base(8, 4.6)
    x, m10, mar = curves(p25, p25.score_fwd)
    _, d10, dar = curves(p25, p25.di)
    k = int(np.ceil(0.1 * len(p25)))
    top = p25.sort_values("score_fwd", ascending=False).head(k)
    headline(fig, f"2025년 검증: 위험 상위 10%에 100ha+ 대형산불 {top.y_100ha.sum()}건이 모두 들어왔다",
             f"2022~2024년 데이터로만 학습한 모델을 2025년 산불 {len(p25)}건에 적용 · 신고 시점 점수 상위 10%({k}건)에\n"
             f"10ha 이상 {top.y_10ha.sum()}/{p25.y_10ha.sum()}건, 100ha 이상 {top.y_100ha.sum()}/{p25.y_100ha.sum()}건, 피해면적 {100 * top.area_ha.sum() / p25.area_ha.sum():.1f}% 포함 (산불위험지수 단독은 피해면적 {100 * dar[np.searchsorted(x, 0.1)]:.0f}%)")
    for j, (ttl, m, dd) in enumerate([("10ha 이상 산불 포착률", m10, d10), ("피해면적 포착률", mar, dar)]):
        ax = fig.add_axes([0.08 + j * 0.47, 0.15, 0.4, 0.52])
        ax.plot(100 * x, 100 * x, color=GRAYMARK, lw=1.2, zorder=1)
        ax.plot(100 * x, 100 * dd, color=ORANGE, lw=2, zorder=2, solid_capstyle="round")
        ax.plot(100 * x, 100 * m, color=BLUE, lw=2, zorder=3, solid_capstyle="round")
        ax.axvline(10, color=MUTED, lw=0.8, zorder=0)
        i10 = np.searchsorted(x, 0.1)
        for arr, col, nm in ((m, BLUE, "결합 모델"), (dd, ORANGE, "위험지수 단독")):
            v = 100 * arr[i10]
            ax.scatter([100 * x[i10]], [v], s=26, color=col, zorder=4, edgecolors=SURF, linewidths=1.5)
            if v > 88:      # 곡선이 위에 붙어 있으면 아래쪽
                ax.text(11.2, v - 10, f"{nm} {v:.0f}%", fontsize=8, color=INK, fontweight="bold")
            elif v < 12:    # 바닥에 붙어 있으면 위쪽
                ax.text(11.2, v + 5, f"{nm} {v:.0f}%", fontsize=8, color=INK, fontweight="bold")
            elif col == BLUE:  # 왼쪽 위 (곡선이 아래에 있음)
                ax.text(9.2, v + 4, f"{nm} {v:.0f}%", fontsize=8, color=INK, fontweight="bold", ha="right")
            else:
                ax.text(11.2, v - 8, f"{nm} {v:.0f}%", fontsize=8, color=INK, fontweight="bold")
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 102)
        ax.set_xlabel("우선 대응 대상 비율(점수 상위 %)", fontsize=8, color=MUTED)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_title(ttl, fontsize=10.5)
        clean(ax, grid="both")
        ax.spines["left"].set_visible(True)
        if j == 0:
            ax.plot([], [], color=BLUE, lw=2, label="결합 모델(산림청+기상청)")
            ax.plot([], [], color=ORANGE, lw=2, label="산불위험지수 단독")
            ax.plot([], [], color=GRAYMARK, lw=1.2, label="무작위")
            ax.legend(loc="upper left", frameon=False, fontsize=7.8)
    footer(fig, "포착률 = 점수 상위 x% 산불에 포함된 비율 · 결합 모델 = 로지스틱 회귀 + LightGBM 순위 평균 · 입력은 신고 시점 정보만")
    fig.savefig(FIG / "04_forward_2025.png")
    plt.close(fig)


# ------------------------------------------------------------------ 5. 2025.3.22 사례
def fig5_case(p, fires):
    p25 = p[p.year == 2025].copy()
    thr = p25.score_fwd.quantile(0.9)
    day = p25[p25.date == "2025-03-22"].copy()
    day["flag"] = day.score_fwd >= thr
    fig = fig_base(8, 4.8)
    big = day[day.area_ha >= 10]
    headline(fig, f"2025.3.22 동시다발 {len(day)}건: 모델이 '위험'으로 고른 {day.flag.sum()}건에 10ha 이상 {big.flag.sum()}건 모두",
             f"위험으로 고르지 않은 {(~day.flag).sum()}건은 모두 {day[~day.flag].area_ha.max():.0f}ha 이하 · 의성 안평 산불은 신고 즉시 '위험' → 동원령 1단계는 101분 뒤")
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.58])
    t = day.occ.dt.hour + day.occ.dt.minute / 60
    size = 18 + 260 * np.sqrt(day.area_ha / day.area_ha.max())
    col = np.where(day.area_ha >= 10, ORANGE, GRAYMARK)
    ax.axhspan(thr, 1.0, color="#eaf2fc", zorder=0)
    ax.axhline(thr, color=BLUE, lw=0.9, zorder=1)
    ax.text(19.9, thr + 0.004, "위험 상위 10% 기준선", color=INK2, fontsize=7.5, ha="right", va="bottom")
    ax.scatter(t, day.score_fwd, s=size, c=col, edgecolors=SURF, linewidths=1.5, zorder=3)
    ann = {"안평": ("의성 안평 46,575ha\n11:24 신고 → 1단계 동원령 13:05", (-1.8, -0.17)),
           "안계": ("의성 안계 52,707ha", (0.6, -0.12)),
           "온양": ("울주 온양 1,117ha\n12:12 신고 → 1단계 15:00", (-0.3, -0.25)),
           "한림": ("김해 한림 97ha", (1.2, 0.012))}
    for r, tt in zip(day.itertuples(), t):
        e = r.emd_raw if isinstance(r.emd_raw, str) else ""
        if e in ann:
            txt, (dx, dy) = ann[e]
            ax.annotate(txt, (tt, r.score_fwd), xytext=(tt + dx, r.score_fwd + dy), fontsize=7.6, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), va="center")
    ax.set_xlim(9.5, 20.2)
    ax.set_ylim(0.55, 1.01)
    ax.set_xticks(range(10, 21, 2), [f"{h}시" for h in range(10, 21, 2)])
    ax.set_ylabel("대형화 위험 점수(2025년 내 순위 기준)", fontsize=8, color=MUTED)
    ax.set_xlabel("신고(발생) 시각", fontsize=8, color=MUTED)
    clean(ax, grid="y")
    ax.scatter([], [], s=60, color=ORANGE, label="10ha 이상으로 확산")
    ax.scatter([], [], s=30, color=GRAYMARK, label="10ha 미만")
    ax.legend(loc="lower right", frameon=False, fontsize=7.8)
    footer(fig, "점 크기 = 최종 피해면적 · 점수는 2022~2024년으로만 학습한 모델의 2025년 예측(사후 정보 미사용) · 동원령: 산림청_산불 단계별 동원력 이력")
    fig.savefig(FIG / "05_case_20250322.png")
    plt.close(fig)


# ------------------------------------------------------------------ 6. 모델이 본 신호(SHAP)
def fig6_shap():
    imp = pd.read_csv(PROC / "shap_importance.csv")
    fig = fig_base(8, 4.6)
    g = imp.groupby("group").mean_abs_shap.sum().sort_values()
    share = g / g.sum()
    headline(fig, "모델이 본 신호: 바람·건조(기상청)와 산림청 위험지수·예보, 지역 특성",
             "평균 |SHAP| 기준 기여도 — 데이터 출처별 합계(왼쪽)와 상위 변수(오른쪽)")
    ax = fig.add_axes([0.25, 0.14, 0.27, 0.56])
    ax.barh(range(len(g)), 100 * share, height=0.5, color=BLUE, zorder=2)
    for i, v in enumerate(share):
        ax.text(100 * v + 0.8, i, f"{100 * v:.0f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(g)), g.index, fontsize=8)
    ax.set_xlim(0, 100 * share.max() * 1.3)
    ax.set_xticks([])
    clean(ax, grid=None)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("출처별 기여 비중", fontsize=10)
    ax2 = fig.add_axes([0.73, 0.14, 0.24, 0.56])
    top = imp.head(10).iloc[::-1]
    ax2.barh(range(len(top)), top.mean_abs_shap, height=0.5, color=BLUE, zorder=2)
    ax2.set_yticks(range(len(top)), top.label, fontsize=7.8)
    ax2.set_xticks([])
    clean(ax2, grid=None)
    ax2.spines["bottom"].set_visible(False)
    ax2.set_title("상위 10개 변수", fontsize=10)
    footer(fig, "LightGBM(단조 제약) 전체 데이터 학습, 시드 5개 평균 · 위치(위·경도)는 지형·임상·바람길 등 지역 특성의 대리 변수")
    fig.savefig(FIG / "06_shap.png")
    plt.close(fig)


# ------------------------------------------------------------------ 7. 지도
def sido_shapes():
    from shapely.geometry import shape
    from shapely.ops import unary_union
    g = json.load(open(glob(str(RAW / "adm" / "*.geojson"))[0], encoding="utf-8"))
    by = {}
    for f in g["features"]:
        by.setdefault(f["properties"]["sido"], []).append(shape(f["geometry"]).buffer(0))
    return {k: unary_union(v).simplify(0.004) for k, v in by.items()}


def draw_geom(ax, geom, **kw):
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    for p in polys:
        if p.area < 0.0004:
            continue
        x, y = p.exterior.xy
        ax.fill(x, y, **kw)


def fig7_map(d):
    from load import load_resources
    res = load_resources()
    shapes = sido_shapes()
    fig = fig_base(8, 6.2)
    big = d[d.area_ha >= 10]
    headline(fig, "10ha 이상으로 커진 산불은 동쪽·남동쪽 산림지대에 몰린다",
             f"2022.1~2025.9 산불 {len(d):,}건(회색 점) · 10ha 이상 {len(big)}건(주황, 크기 = 피해면적) · 산림항공관리소(파랑 삼각형)", y=0.965)
    ax = fig.add_axes([0.0, 0.05, 0.66, 0.8])
    for geom in shapes.values():
        draw_geom(ax, geom, fc=LAND, ec=BASE, lw=0.4, zorder=1)
    ax.scatter(d.lon, d.lat, s=2.5, color=MUTED, alpha=0.55, lw=0, zorder=2)
    s = 12 + 900 * np.sqrt(big.area_ha / d.area_ha.max())
    ax.scatter(big.lon, big.lat, s=s, color=ORANGE, alpha=0.85, edgecolors=SURF, linewidths=1.2, zorder=3)
    ax.scatter(res["heli"].lon, res["heli"].lat, marker="^", s=34, color=BLUE, edgecolors=SURF, linewidths=1.0, zorder=4)
    ax.set_xlim(124.4, 131.6)
    ax.set_ylim(33.1, 38.7)
    ax.set_aspect(1 / np.cos(np.radians(36)))
    ax.axis("off")
    top = big.nlargest(6, "area_ha")
    uis = top[top.addr.fillna("").str.contains("의성군")]
    labels = [(f"2025.3 의성 안평·안계\n(발화점 2곳) {uis.area_ha.sum():,.0f}ha", uis.lon.mean(), uis.lat.mean(), (129.75, 35.95), "left")]
    # 나머지 상위 산불: 바다 쪽 고정 위치에 라벨
    spots = {"울진군": ((129.75, 36.95), "left"), "강릉시": ((129.45, 37.95), "left"),
             "산청군": ((127.2, 34.2), "left"), "홍성군": ((125.75, 37.05), "right")}
    for r in top[~top.index.isin(uis.index)].itertuples():
        key = next((k for k in spots if isinstance(r.addr, str) and k in r.addr), None)
        if key:
            (tx, ty), ha = spots[key]
            labels.append((f"{r.occ.year}.{r.occ.month} {place(r, with_date=False).split(' ', 1)[1]}\n{r.area_ha:,.0f}ha", r.lon, r.lat, (tx, ty), ha))
    for txt, lo, la, (tx, ty), ha in labels:
        ax.annotate(txt, (lo, la), xytext=(tx, ty), fontsize=7, color=INK, ha=ha, va="center", linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=2, shrinkB=2))
    # 오른쪽: 시도별 10ha+ 비율
    ax2 = fig.add_axes([0.72, 0.12, 0.25, 0.66])
    t = d.groupby("g_sido").agg(n=("y_10ha", "size"), k=("y_10ha", "sum"))
    t = t[t.n >= 40]
    t["p"] = t.k / t.n
    t = t.sort_values("p")
    ax2.barh(range(len(t)), 100 * t.p, height=0.55, color=ORANGE, zorder=2)
    for i, (p_, k, n) in enumerate(zip(t.p, t.k, t.n)):
        ax2.text(100 * p_ + 0.3, i, f"{100 * p_:.1f}% ({k}/{n})", va="center", fontsize=7, color=INK)
    ax2.set_yticks(range(len(t)), t.index, fontsize=8)
    ax2.set_xticks([])
    ax2.set_xlim(0, 100 * t.p.max() * 1.7)
    clean(ax2, grid=None)
    ax2.spines["bottom"].set_visible(False)
    ax2.set_title("시도별 10ha 이상 확산 비율", fontsize=10)
    fig.text(0.72, 0.075, "산불 40건 이상 시도", fontsize=7, color=MUTED)
    footer(fig, "위치 = 발생 읍면동(행정동) 대표점 · 경계: 통계청 SGIS 행정동 경계(vuski/admdongkor, CC BY 4.0) · 산림항공관리소: 산림청_진화대원대기장소")
    fig.savefig(FIG / "07_map.png")
    plt.close(fig)


# ------------------------------------------------------------------ 8. 데이터 품질
def fig8_quality():
    Q = json.load(open(PROC / "quality_summary.json", encoding="utf-8"))
    s, h, r, j, f = Q["stats"], Q["history"], Q["risk"], Q["join"], Q["forecast"]
    items = [
        ("공통 산불 ID 없음", f"통계·위험지수·동원령에 ID가 없어 지명·시각으로 추정 결합({j['matched']:,}/{j['total']:,}건)"),
        ("데이터사전 ↔ 파일 불일치", f"사전 {s['dict_cols']}개 컬럼 중 법정동코드·오류플래그 등 {len(s['dict_cols_missing_in_file'])}개가 실제 파일에 없음"),
        ("포털 메타데이터 오기", f"수록기간 '{s['portal_meta']['temporalCoverage']}'·전체 행 '{s['portal_meta']['전체 행']}' ↔ 실제 2025.9까지 {s['rows']:,}행"),
        ("진화 종료 시각 불일치", f"통계 '진화종료'와 이력 '진화완료'가 6시간 넘게 다른 산불 {j['end_mismatch_gt6h']}건(24시간+ {j['end_mismatch_gt24h']}건)"),
        ("시각 오류", f"종료<발생 {s['err_reversed']}건, 종료연도 {'·'.join(map(str, s['err_year_values']))}년, 이력 진화완료<진화시작 {h['end_before_start']}건"),
        ("원인 코드 미활용", f"'기타(직접입력)' {s['cause_etc_direct']:,}건 중 {s['cause_etc_reclassifiable']:,}건이 표준 원인으로 분류 가능(자유기재 {s['cause_etc_unique_texts']}종)"),
        ("위험지수 결측", f"결측 {100 * r['missing']:.1f}% (2022년 {100 * r['missing_by_year']['2022']:.0f}% → 2025년 {100 * r['missing_by_year']['2025']:.0f}%), 소형 산불에 결측 집중"),
        ("일출·일몰 위치 미반영", f"같은 날이면 전국 동일 값 → 위치별 계산과 최대 {h['sunset_dev_max_min']:.0f}분 차이"),
        ("예보 시도명 혼재", f"17개 시도가 {f['sido_variants']}가지 표기(강원/강원도/강원특별자치도 …)"),
        ("예보 등급 기준 불일치", f"경보 기준 충족 {f['meets_warn_but_jueuibo']}건이 '주의보' · '경보' 1건은 실효습도 {f['the_only_warn'][0]['eff_hum']}%"),
    ]
    fig = fig_base(8, 5.4)
    headline(fig, "활용하며 확인한 데이터 품질 이슈 — 모두 원본 파일에서 재계산 가능",
             "산림청 산불 데이터를 결합하며 직접 부딪힌 문제 (src/quality.py → reports/data_quality.md, 17건 중 10건)", y=0.965)
    y0 = 0.80
    for i, (k, v) in enumerate(items):
        y = y0 - i * 0.071
        fig.add_artist(plt.Line2D([0.035, 0.965], [y + 0.03, y + 0.03], color=GRID, lw=0.6))
        fig.text(0.035, y, f"{i + 1:>2}", fontsize=9, color=MUTED, va="center")
        fig.text(0.07, y, k, fontsize=9, color=INK, fontweight="bold", va="center")
        fig.text(0.31, y, v, fontsize=8.4, color=INK2, va="center")
    footer(fig, "자료: 공공데이터포털 산림청·국립산림과학원 파일데이터(2026.11.13 내려받음)")
    fig.savefig(FIG / "08_data_quality.png")
    plt.close(fig)


def poster(names, out="00_summary_poster.png"):
    from PIL import Image
    ims = [Image.open(FIG / n).convert("RGB") for n in names]
    w = max(i.width for i in ims)
    h = sum(i.height for i in ims) + 24 * (len(ims) - 1)
    canvas = Image.new("RGB", (w, h), (233, 232, 226))
    y = 0
    for im in ims:
        canvas.paste(im, (0, y))
        y += im.height + 24
    canvas.save(FIG / out, optimize=True)


def main():
    d = pd.read_csv(PROC / "fires.csv", parse_dates=["occ", "date"])
    p = pd.read_csv(PROC / "predictions.csv", parse_dates=["occ", "date"])
    fig0_pipeline(d)
    fig1_concentration(d)
    fig2_concurrency(d)
    fig3_signals(d)
    fig4_forward(p, d)
    fig5_case(p, d)
    fig6_shap()
    fig7_map(d)
    fig8_quality()
    poster(["00_pipeline.png", "01_concentration.png", "02_concurrency.png", "04_forward_2025.png", "05_case_20250322.png"])
    # 첨부가 1개만 될 때를 위한 PDF 묶음 (제출 순서)
    from PIL import Image
    order = ["00_pipeline", "01_concentration", "02_concurrency", "03_signals", "04_forward_2025",
             "05_case_20250322", "07_map", "06_shap", "08_data_quality"]
    pages = [Image.open(FIG / f"{n}.png").convert("RGB") for n in order]
    pages[0].save(FIG / "all_figures.pdf", save_all=True, append_images=pages[1:], resolution=200)
    print("saved:", sorted(x.name for x in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
