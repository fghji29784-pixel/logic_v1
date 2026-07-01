# ─────────────────────────────────────────────
#  parser.py  —  KSS / TEMP_DATA / 공정 데이터 파싱
# ─────────────────────────────────────────────

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from constants import (
    DEVICE_OFFSET, DEVICE_CELL_RANGE,
    TOTAL_CELLS, TRAY_ROWS, TRAY_COLS, NUM_LAYERS,
    PROCESS_COL_TRAY_ID, PROCESS_COL_CELL_NO,
    PROCESS_COL_OCV, PROCESS_COL_DOCV, PROCESS_COL_DOCV_FALLBACK,
    PROCESS_COL_GRADE, PROCESS_COL_CELL_ID, PROCESS_COL_LOT_ID,
)


# ══════════════════════════════════════════════
#  유틸리티
# ══════════════════════════════════════════════

def get_layer(cell_no: int) -> int:
    """셀 번호(1~144) → 트레이 레이어(1=가장자리, 6=중앙)"""
    row = (cell_no - 1) // TRAY_COLS + 1
    col = (cell_no - 1) % TRAY_COLS + 1
    return int(min(row, TRAY_ROWS + 1 - row, col, TRAY_COLS + 1 - col))


def cell_no_from(device_no: int, channel_no: int) -> int:
    """설비번호 + 채널번호 → 셀 번호(1~144)"""
    return DEVICE_OFFSET[device_no] + channel_no


def detect_device_no(filename: str) -> int | None:
    """파일명에서 설비번호(1~5) 추출. 파일명 끝에서부터 역순으로 탐색."""
    stem = Path(filename).stem
    parts = stem.split('_')
    for part in reversed(parts):
        if re.fullmatch(r'[1-5]', part):
            return int(part)
    return None


def _split_line(line: str) -> list[str]:
    """탭, 콤마, 또는 파이프(`|`)로 분리 후 공백 제거"""
    if '\t' in line:
        sep = '\t'
    elif ',' in line:
        sep = ','
    else:
        sep = '|'
    return [p.strip() for p in line.split(sep)]


def _read_text(filepath: str) -> str:
    """UTF-8 우선, 실패 시 EUC-KR로 파일 읽기"""
    for enc in ('utf-8', 'euc-kr', 'cp949'):
        try:
            return Path(filepath).read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return Path(filepath).read_text(encoding='utf-8', errors='replace')


# ══════════════════════════════════════════════
#  KSS 파일 파싱
# ══════════════════════════════════════════════

def _split_sections(content: str) -> dict[str, list[str]]:
    """Cat XXX, Cat_XXX, or [Cat XXX] 섹션별로 분리"""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in content.splitlines():
        line = raw.strip()
        m = re.match(r'^\[?Cat[\s_]+(.+?)\]?$', line)
        if m:
            current = m.group(1).strip()
            sections[current] = []
        elif current is not None and line:
            sections[current].append(line)
    return sections


def _line_key_value(line: str) -> tuple[str, str] | None:
    """Colon, tab, or pipe separated metadata row."""
    if ':' in line:
        k, v = line.split(':', 1)
        return k.strip(), v.strip()

    parts = _split_line(line)
    if len(parts) >= 2 and parts[0]:
        return parts[0].strip(), parts[1].strip()

    return None


def _parse_app_info(lines: list[str]) -> dict:
    out = {}
    for line in lines:
        kv = _line_key_value(line)
        if kv is not None:
            k, v = kv
            out[k] = v
    return out


def _parse_global(lines: list[str]) -> dict:
    out = {'test_duration': 15, 'tint': 10}
    for line in lines:
        kv = _line_key_value(line)
        if kv is None:
            continue
        k, v = kv
        if k == 'TestDuration':
            try:
                out['test_duration'] = int(float(v))
            except ValueError:
                pass
        elif k == 'Tint':
            try:
                out['tint'] = int(float(v))
            except ValueError:
                pass
    return out


def _parse_per_chan(lines: list[str]) -> dict[int, dict]:
    """채널별 {enabled, rwiring} 파싱"""
    channels: dict[int, dict] = {}
    if not lines:
        return channels

    header_idx = None
    for i, line in enumerate(lines):
        parts = _split_line(line)
        lowered = [p.lower() for p in parts]
        if any('channel' in p for p in lowered):
            header_idx = i
            break

    if header_idx is None:
        return channels

    headers = _split_line(lines[header_idx])
    # 컬럼 인덱스 — or 연산자는 0(첫 번째 컬럼)을 falsy로 처리하므로 명시적 None 체크
    def idx(keyword):
        for i, h in enumerate(headers):
            if keyword.lower() in h.lower():
                return i
        return None

    i_ch = idx('Channel');      i_ch = 0 if i_ch is None else i_ch
    i_en = idx('IsChanEnabled')
    if i_en is None:
        i_en = idx('IsChanCheck')
    if i_en is None:
        i_en = idx('Enabled')
    i_en = 1 if i_en is None else i_en
    i_rw = idx('Rwiring')
    if i_rw is None:
        i_rw = idx('Rwire')

    for line in lines[header_idx + 1:]:
        parts = _split_line(line)
        if len(parts) < 2:
            continue
        try:
            ch      = int(parts[i_ch])
            enabled = parts[i_en].strip().upper() in ('TRUE', '1', 'Y', 'YES')
            rwiring = float(parts[i_rw]) if (i_rw is not None and i_rw < len(parts)) else 0.0
            channels[ch] = {'enabled': enabled, 'rwiring': rwiring}
        except (ValueError, IndexError):
            continue
    return channels


def _parse_measurements(lines: list[str]) -> pd.DataFrame | None:
    """시계열 측정값 파싱 → DataFrame"""
    if not lines:
        return None

    header_idx = None
    for i, line in enumerate(lines):
        parts = _split_line(line)
        lowered = [p.lower() for p in parts]
        if lowered and lowered[0] == 'time' and any(re.fullmatch(r'[iv]\d+', p) for p in lowered[1:]):
            header_idx = i
            break

    if header_idx is None:
        return None

    headers = _split_line(lines[header_idx])
    rows = []
    for line in lines[header_idx + 1:]:
        parts = _split_line(line)
        if parts:
            rows.append(parts)

    if not rows:
        return None

    # 행마다 길이가 다를 수 있으므로 최대 길이 기준
    max_len = max(len(r) for r in rows)
    headers = (headers + [''] * max_len)[:max_len]

    df = pd.DataFrame(rows, columns=headers)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def parse_kss_file(filepath: str) -> dict:
    """
    KSS 파일 파싱.
    반환:
      serial_num   : str
      test_duration: int (분)
      tint         : int (초)
      channels     : {ch_no: {enabled, rwiring}}
      measurements : pd.DataFrame  (t, I1..I32, V1..V32)
    """
    content  = _read_text(filepath)
    sections = _split_sections(content)

    app    = _parse_app_info(sections.get('AppInfo', []))
    glb    = _parse_global(sections.get('TestSetupGlobal', []))
    chans  = _parse_per_chan(sections.get('TestSetupPerChan', []))
    meas   = _parse_measurements(sections.get('Measurements', []))

    return {
        'serial_num'   : app.get('SerialNum'),
        'test_duration': glb['test_duration'],
        'tint'         : glb['tint'],
        'channels'     : chans,
        'measurements' : meas,
    }


# ══════════════════════════════════════════════
#  TEMP_DATA 파싱
# ══════════════════════════════════════════════

def parse_temp_data(filepath: str) -> pd.DataFrame:
    """
    TEMP_DATA 파일 파싱.
    반환: DataFrame with columns [t_sec, T_1, ..., T_144]
    """
    fp = Path(filepath)
    try:
        if fp.suffix.lower() in ('.xlsx', '.xls'):
            df = pd.read_excel(fp, header=0)
        else:
            df = pd.read_csv(fp, sep=None, engine='python', header=0)

        n_cols = len(df.columns)
        df.columns = ['t_sec'] + [f'T_{i}' for i in range(1, n_cols)]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    except Exception as e:
        print(f'[TEMP_DATA 파싱 오류] {fp.name}: {e}')
        return pd.DataFrame()


# ══════════════════════════════════════════════
#  트레이 폴더 전체 파싱
# ══════════════════════════════════════════════

def parse_tray_folder(folder_path: str, n_minutes: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    트레이 폴더 1개 파싱.
    반환: (df_meta, df_ts)

    df_meta  — 셀당 1행 (144행 이하)
    df_ts    — 셀×시간 시계열
    """
    folder = Path(folder_path)
    _name = folder.name
    _m = re.search(r'CFDD\S+', _name, re.IGNORECASE)
    tray_id = _m.group(0).upper() if _m else re.sub(r'^\d+\.\s*', '', _name).strip() or _name

    # ── KSS 파일 탐색 ──
    kss_files = list(folder.glob('*.kss')) + list(folder.glob('*.KSS'))
    device_kss: dict[int, str] = {}
    for f in kss_files:
        dev = detect_device_no(f.name)
        if dev is not None:
            device_kss[dev] = str(f)

    # ── TEMP 파일 탐색 (구: *TEMP_DATA*, 신: *TEMP*) ──
    temp_files = list(folder.glob('*TEMP_DATA*')) or list(folder.glob('*TEMP*'))
    temp_df = parse_temp_data(str(temp_files[0])) if temp_files else pd.DataFrame()

    meta_rows: list[dict] = []
    ts_rows: list[dict]   = []

    for device_no, kss_path in sorted(device_kss.items()):
        kss  = parse_kss_file(kss_path)
        meas = kss['measurements']
        if meas is None or meas.empty:
            continue

        t_col_name = meas.columns[0]   # 첫 번째 컬럼 = 시간(초)
        t_sec_arr  = meas[t_col_name].values

        # separation_curve 가 5~30분 범위를 돌 수 있도록 모든 n값 사전 계산
        N_ALL = range(5, 31)

        active_channels = [ch for ch, info in kss['channels'].items() if info['enabled']]

        for ch in active_channels:
            cell_no = cell_no_from(device_no, ch)
            if not (1 <= cell_no <= TOTAL_CELLS):
                continue

            # 전류·전압 컬럼 (I1, V1 형식)
            i_col = f'I{ch}'
            v_col = f'V{ch}'
            if i_col not in meas.columns or v_col not in meas.columns:
                continue

            cur_arr = meas[i_col].values.astype(float)
            vol_arr = meas[v_col].values.astype(float)

            v_init  = vol_arr[0]  * 1000.0 if len(vol_arr) > 0 else np.nan   # V→mV
            v_final = vol_arr[-1] * 1000.0 if len(vol_arr) > 0 else np.nan   # V→mV

            # 온도
            t_key  = f'T_{cell_no}'
            t_init = t_final = np.nan
            if not temp_df.empty and t_key in temp_df.columns:
                temps   = temp_df[t_key].values
                t_init  = float(temps[0])  if len(temps) > 0 else np.nan
                t_final = float(temps[-1]) if len(temps) > 0 else np.nan

            row: dict = {
                'tray_id'  : tray_id,
                'cell_no'  : cell_no,
                'device_no': device_no,
                'channel_no': ch,
                'rwiring'  : kss['channels'][ch]['rwiring'],
                'v_init'   : v_init,
                'v_final'  : v_final,
                'delta_v'  : (v_init - v_final)   # ΔV = V_init − V_final (OCV drop)
                             if not (np.isnan(v_init) or np.isnan(v_final)) else np.nan,
                't_init'   : t_init,
                't_final'  : t_final,
                'delta_t'  : (t_final - t_init)
                             if not (np.isnan(t_init) or np.isnan(t_final)) else np.nan,
                'layer'    : get_layer(cell_no),
            }

            # 모든 n에 대해 i_nmin · slope 계산 (separation_curve 용)
            for n in N_ALL:
                n_sec_n = n * 60
                idx_n   = min(int(np.searchsorted(t_sec_arr, n_sec_n)), len(t_sec_arr) - 1)
                i_val   = cur_arr[idx_n] if idx_n < len(cur_arr) else np.nan
                slp     = (cur_arr[idx_n] - cur_arr[0]) / n_sec_n if n_sec_n > 0 else np.nan
                row[f'i_{n}min']    = i_val
                row[f'slope_0_{n}'] = slp

            meta_rows.append(row)

            # 시계열
            temp_arr = temp_df[t_key].values if (not temp_df.empty and t_key in temp_df.columns) else None
            for j, (t, i_val, v_val) in enumerate(zip(t_sec_arr, cur_arr, vol_arr)):
                temp_val = np.nan
                if temp_arr is not None and j < len(temp_arr):
                    temp_val = float(temp_arr[j])
                ts_rows.append({
                    'tray_id'   : tray_id,
                    'cell_no'   : cell_no,
                    't_sec'     : float(t),
                    'current_A' : float(i_val),
                    'voltage_V' : float(v_val),
                    'temp_C'    : temp_val,
                })

    df_meta = pd.DataFrame(meta_rows)
    df_ts   = pd.DataFrame(ts_rows)
    return df_meta, df_ts


# ══════════════════════════════════════════════
#  공정 데이터 파싱 및 병합
# ══════════════════════════════════════════════

def parse_process_data(filepath: str) -> pd.DataFrame:
    """공정 데이터 파일 파싱 (1행 = 컬럼명)"""
    fp = Path(filepath)
    try:
        if fp.suffix.lower() in ('.xlsx', '.xls'):
            df = pd.read_excel(fp, header=0)
        else:
            df = pd.read_csv(fp, sep=None, engine='python', header=0)
        return df
    except Exception as e:
        print(f'[공정 데이터 파싱 오류] {fp.name}: {e}')
        return pd.DataFrame()


def _find_col(df: pd.DataFrame, keyword: str) -> str | None:
    """컬럼명에 keyword가 포함된 첫 번째 컬럼 반환"""
    for col in df.columns:
        if keyword in str(col):
            return col
    return None


def merge_process_data(df_meta: pd.DataFrame,
                       process_df: pd.DataFrame) -> pd.DataFrame:
    """
    df_meta + 공정 데이터를 TRAY ID & CELL NO 기준으로 Left Join.
    공정 데이터가 없으면 df_meta 그대로 반환.
    """
    if process_df.empty or df_meta.empty:
        return df_meta

    # 대표 매칭 컬럼 탐색 (TRAY ID_* / CELL NO_* 중 하나)
    tray_col = _find_col(process_df, 'TRAY ID')
    cell_col = _find_col(process_df, 'CELL NO')

    if tray_col is None or cell_col is None:
        print('[경고] 공정 데이터에서 TRAY ID 또는 CELL NO 컬럼을 찾을 수 없습니다.')
        return df_meta

    process_df = process_df.copy()

    # dOCV 직접 컬럼이 없으면 (OCV1 - OCV3) 로 대체 계산 → PROCESS_COL_DOCV 로 사용
    if PROCESS_COL_DOCV not in process_df.columns:
        c1, c3 = PROCESS_COL_DOCV_FALLBACK
        if c1 in process_df.columns and c3 in process_df.columns:
            process_df[PROCESS_COL_DOCV] = (
                pd.to_numeric(process_df[c1], errors='coerce')
                - pd.to_numeric(process_df[c3], errors='coerce')
            )
            print(f'[정보] dOCV 컬럼이 없어 ({c1} - {c3}) 로 대체 계산했습니다.')

    # 필요한 컬럼만 추출
    want_cols = [tray_col, cell_col]
    for col in PROCESS_COL_OCV.values():
        if col in process_df.columns:
            want_cols.append(col)
    for col in [PROCESS_COL_DOCV, PROCESS_COL_GRADE,
                PROCESS_COL_CELL_ID, PROCESS_COL_LOT_ID]:
        if col in process_df.columns:
            want_cols.append(col)

    proc = process_df[want_cols].copy()
    proc = proc.rename(columns={tray_col: '_tray_key', cell_col: '_cell_key'})
    proc['_tray_key'] = proc['_tray_key'].astype(str).str.strip().str.upper()
    proc['_cell_key'] = pd.to_numeric(proc['_cell_key'], errors='coerce')

    meta = df_meta.copy()
    meta['_tray_key'] = meta['tray_id'].astype(str).str.strip().str.upper()
    meta['_cell_key'] = pd.to_numeric(meta['cell_no'], errors='coerce')

    merged = meta.merge(proc, on=['_tray_key', '_cell_key'], how='left')
    merged = merged.drop(columns=['_tray_key', '_cell_key'])

    return merged


# ══════════════════════════════════════════════
#  다중 트레이 폴더 일괄 파싱
# ══════════════════════════════════════════════

def parse_multiple_trays(folder_paths: list[str],
                         process_filepath: str | None,
                         n_minutes: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    여러 트레이 폴더 일괄 파싱 → 통합 (df_meta, df_ts).
    공정 데이터가 있으면 df_meta에 병합.
    """
    all_meta: list[pd.DataFrame] = []
    all_ts:   list[pd.DataFrame] = []

    for folder in folder_paths:
        dm, dt = parse_tray_folder(folder, n_minutes=n_minutes)
        if not dm.empty:
            all_meta.append(dm)
        if not dt.empty:
            all_ts.append(dt)

    df_meta = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    df_ts   = pd.concat(all_ts,   ignore_index=True) if all_ts   else pd.DataFrame()

    if process_filepath and not df_meta.empty:
        proc_df = parse_process_data(process_filepath)
        df_meta = merge_process_data(df_meta, proc_df)

    return df_meta, df_ts
