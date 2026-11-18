"""공모 폼 '활용 내용(추가)' 첨부용 이미지 3장 (png, 장당 10MB 이하).

사진 1: 데이터 결합과 분석 / 사진 2: 트리아지 모델 검증 / 사진 3: 데이터 품질 점검 + 코드 저장소
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import FIG, PROC
from figures import BLUE, GRID, INK, INK2, MUTED, ORANGE, SURF, fig_base, footer, headline

OUT = FIG / "submit"
OUT.mkdir(exist_ok=True)
REPO = "github.com/hwaAAnn/Contest-Forest_Wildfire_Golden_Time"


def header(text, sub, name):
    fig = plt.figure(figsize=(8, 0.85), dpi=200)
    fig.patch.set_facecolor("#20201f")
    fig.text(0.035, 0.62, text, fontsize=15, fontweight="bold", color="#ffffff", va="center")
    fig.text(0.035, 0.24, sub, fontsize=8.5, color="#c3c2b7", va="center")
    path = OUT / f"_{name}.png"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def repo_panel():
    fig = fig_base(8, 5.6)
    headline(fig, "코드·결과 전체 공개 — 내려받기부터 그림까지 명령 5줄로 재현", REPO, y=0.965)
    rows = [("python src/download.py", "공공데이터포털 파일 8종(로그인 불필요) · 기상청 API허브 ASOS 일자료 · 행정동 경계"),
            ("python src/build.py", "결합·지오코딩·기상·예보·동시다발 변수 → fires.csv (2,020건)"),
            ("python src/model.py", "트리아지 모델 · 연도 교차검증 · 2025 시간 외 검증 · SHAP"),
            ("python src/quality.py", "데이터 품질 점검 17항목 → reports/data_quality.md"),
            ("python src/figures.py", "제출용 그림 생성")]
    y = 0.79
    for cmd, desc in rows:
        fig.add_artist(plt.Line2D([0.035, 0.965], [y + 0.035, y + 0.035], color=GRID, lw=0.6))
        fig.text(0.035, y, cmd, fontsize=9, family="Consolas", color=INK, va="center")
        fig.text(0.35, y, desc, fontsize=8.4, color=INK2, va="center")
        y -= 0.07
    # 성능 비교 표 (2025년 시간 외 검증)
    f = pd.read_csv(PROC / "metrics_forward2025.csv")
    pick = {"산불위험지수(산림청)": "산불위험지수 단독", "실효습도(기상청, 낮을수록 위험)": "실효습도 단독",
            "최대풍속(기상청)": "최대풍속 단독", "앙상블": "결합 모델(최종)"}
    f = f[f.model.isin(pick)].assign(model=lambda x: x.model.map(pick))
    y0 = 0.36
    fig.text(0.035, y0 + 0.06, "2025년 시간 외 검증 (2022~2024년으로만 학습, 2025년 397건 적용) · 점수 상위 10%가 포착한 비율",
             fontsize=9, color=INK, fontweight="bold", va="center")
    cols = [("", 0.035), ("AUC", 0.36), ("10ha 이상", 0.5), ("100ha 이상", 0.65), ("피해면적", 0.8)]
    for t, x in cols:
        fig.text(x, y0, t, fontsize=8.2, color=MUTED, va="center")
    for i, r in enumerate(f.itertuples()):
        yy = y0 - 0.06 * (i + 1)
        best = r.model.startswith("결합")
        kw = dict(fontsize=8.8, va="center", color=INK if best else INK2, fontweight="bold" if best else "normal")
        fig.add_artist(plt.Line2D([0.035, 0.965], [yy + 0.03, yy + 0.03], color=GRID, lw=0.6))
        fig.text(0.035, yy, r.model, **kw)
        for v, (_, x) in zip([f"{r.auc:.2f}", f"{100 * r._5:.0f}%", f"{100 * r._6:.0f}%", f"{100 * r._7:.1f}%"], cols[1:]):
            fig.text(x, yy, v, **kw)
    footer(fig, "자료: 산림청·국립산림과학원 공공데이터(공공데이터포털), 기상청 API허브 ASOS, 통계청 SGIS 행정동 경계(vuski/admdongkor)")
    path = OUT / "_repo.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def stack(paths, out):
    ims = [Image.open(p).convert("RGB") for p in paths]
    w = max(i.width for i in ims)
    gap = 20
    canvas = Image.new("RGB", (w, sum(i.height for i in ims) + gap * (len(ims) - 1)), (233, 232, 226))
    y = 0
    for im in ims:
        canvas.paste(im, (0, y))
        y += im.height + gap
    canvas.save(out, optimize=True)
    print(out.name, canvas.size, f"{out.stat().st_size / 1e6:.2f}MB")


def main():
    stack([header("① 데이터 결합과 분석", "산림청 산불 데이터 7종 + 기상청 ASOS를 '산불 1건 = 1행'으로 결합 (2,020건, 2022.1~2025.9)", "h1"),
           FIG / "00_pipeline.png", FIG / "01_concentration.png", FIG / "02_concurrency.png", FIG / "03_signals.png"],
          OUT / "사진1_데이터결합_분석.png")
    stack([header("② 신고 즉시 대형화 위험 트리아지 모델", "신고 시점 정보만 사용 · 2022~2024년 학습 → 2025년 시간 외 검증", "h2"),
           FIG / "04_forward_2025.png", FIG / "05_case_20250322.png", FIG / "07_map.png", FIG / "06_shap.png"],
          OUT / "사진2_트리아지모델_검증.png")
    stack([header("③ 데이터 품질 점검과 코드 저장소", f"개선 아이디어의 근거 17건 · {REPO}", "h3"),
           FIG / "08_data_quality.png", repo_panel()],
          OUT / "사진3_품질점검_코드저장소.png")
    for p in OUT.glob("_*.png"):
        p.unlink()


if __name__ == "__main__":
    main()
