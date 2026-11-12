"""원본 데이터 내려받기.

python src/download.py            # 전부
python src/download.py --only datagokr|kma|adm
"""
import argparse
import io
import json
import re
import time
import zipfile
from datetime import date

import pandas as pd
import requests

from config import RAW, DATASETS, WX_START, WX_END, ADM_URL, kma_key

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


# ---------------------------------------------------------------- data.go.kr
def datagokr_file(pk: str, session: requests.Session) -> str:
    """공공데이터포털 파일데이터 1건 다운로드 → data/raw/datagokr/<pk>_<원본파일명>."""
    out = RAW / "datagokr"
    out.mkdir(exist_ok=True)
    page = session.get(f"https://www.data.go.kr/data/{pk}/fileData.do", timeout=30).text
    m = re.search(r"fn_fileDataDown\(\s*'(\d+)'\s*,\s*'([^']+)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'", page)
    if not m:
        raise RuntimeError(f"{pk}: 다운로드 정보를 찾지 못함")
    _, uddi, atch, sn = m.groups()
    r = session.post("https://www.data.go.kr/tcs/dss/selectFileDataDownload.do",
                     data=dict(publicDataPk=pk, publicDataDetailPk=uddi, atchFileId=atch,
                               fileDetailSn=sn or "1", publicDataTyCode="PR0051"), timeout=30)
    j = json.loads(r.text)
    if not j.get("status"):
        raise RuntimeError(f"{pk}: {j.get('error')}")
    info = j["dataSetFileDetailInfo"]
    name = info["publicDataSj"] if info.get("publicDataTyDetailCode") == "DATY05" else info["dataNm"]
    f = session.get("https://www.data.go.kr/cmm/cmm/fileDownload.do",
                    params=dict(atchFileId=j["atchFileId"], fileDetailSn=j["fileDetailSn"], dataNm=name),
                    timeout=180)
    f.raise_for_status()
    cd = f.headers.get("Content-Disposition", "")
    mm = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd)
    fn = requests.utils.unquote(mm.group(1)) if mm else f"{pk}.csv"
    try:
        fn = fn.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    path = out / f"{pk}_{fn}"
    path.write_bytes(f.content)
    if path.suffix.lower() == ".zip":  # 산불통계데이터는 zip(CP949 파일명) 안에 CSV·데이터사전·점자파일
        zdir = out / f"{pk}_unzip"
        zdir.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(f.content)) as z:
            for i in z.infolist():
                if i.is_dir():
                    continue
                try:
                    nm = i.filename.encode("cp437").decode("cp949")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    nm = i.filename
                (zdir / nm.split("/")[-1]).write_bytes(z.read(i))
    return path.name


def download_datagokr():
    s = requests.Session()
    s.headers.update(UA)
    for pk, nm in DATASETS.items():
        try:
            print(f"[data.go.kr] {pk} {nm} -> {datagokr_file(pk, s)}")
        except Exception as e:  # noqa: BLE001
            print(f"[data.go.kr] {pk} 실패: {e}")
        time.sleep(1.0)


# ---------------------------------------------------------------- 기상청 API허브
KMA = "https://apihub.kma.go.kr/api/typ01/url"


def _kma_table(text: str, cols: list[str] | None = None) -> pd.DataFrame:
    rows = [ln.split() for ln in text.splitlines() if ln and not ln.startswith("#")]
    return pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows)


def download_kma():
    key = kma_key()
    out = RAW / "kma"
    out.mkdir(exist_ok=True)
    # 1) ASOS 지점 정보 (위경도)
    tm = WX_END.replace("-", "") + "0900"
    r = requests.get(f"{KMA}/stn_inf.php", params=dict(inf="SFC", stn="", tm=tm, help=1, authKey=key), timeout=60)
    r.encoding = "cp949"
    (out / "stn_inf_sfc.txt").write_text(r.text, encoding="utf-8")
    print("[kma] 지점정보", len(r.text), "bytes")
    # 2) ASOS 일자료 (전 지점, 월 단위)
    months = pd.period_range(WX_START, WX_END, freq="M")
    for p in months:
        f = out / f"sfcdd_{p.strftime('%Y%m')}.txt"
        if f.exists() and f.stat().st_size > 1000:
            continue
        t1 = p.start_time.strftime("%Y%m%d")
        t2 = min(p.end_time.date(), date.fromisoformat(WX_END)).strftime("%Y%m%d")
        for attempt in range(3):
            try:
                r = requests.get(f"{KMA}/kma_sfcdd3.php",
                                 params=dict(tm1=t1, tm2=t2, stn=0, help=0, authKey=key), timeout=120)
                r.encoding = "cp949"
                if "#START7777" in r.text:
                    f.write_text(r.text, encoding="utf-8")
                    break
            except requests.RequestException:
                pass
            time.sleep(3)
        print(f"[kma] {p} {f.stat().st_size if f.exists() else 'FAIL'}")
        time.sleep(0.5)


# ---------------------------------------------------------------- 행정동 경계
def download_adm():
    out = RAW / "adm"
    out.mkdir(exist_ok=True)
    f = out / ADM_URL.split("/")[-1]
    if not f.exists():
        r = requests.get(ADM_URL, timeout=300)
        r.raise_for_status()
        f.write_bytes(r.content)
    print("[adm]", f.name, f.stat().st_size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["datagokr", "kma", "adm"])
    a = ap.parse_args()
    if a.only in (None, "datagokr"):
        download_datagokr()
    if a.only in (None, "adm"):
        download_adm()
    if a.only in (None, "kma"):
        download_kma()
