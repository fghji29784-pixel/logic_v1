from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QListWidget,
    QListWidgetItem, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QGroupBox,
    QTableWidget, QTableWidgetItem, QSplitter,
    QMessageBox, QProgressDialog, QSlider, QLineEdit,
    QHeaderView, QAbstractItemView, QFrame, QScrollArea,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from parser   import parse_multiple_trays, get_layer
from analysis import (
    add_layer_dummies, run_analysis, run_analysis_per_tray,
    separation_curve, confusion_at_threshold, compute_metrics,
    docv_surrogate_analysis,
)
from constants import (
    TOTAL_CELLS, TRAY_ROWS, TRAY_COLS, NUM_LAYERS,
    PROCESS_COL_GRADE, PROCESS_COL_DOCV,
    REF_V_INIT, REF_T_FINAL,
    cell_to_label,
)

# ── matplotlib 한글 폰트 ──────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
from app_state import AppState, AnalysisWorker
from ui_common import PlotCanvas


def _fmt(v, unit: str = '', nd: int = 3) -> str:
    """숫자 → 표시 문자열. NaN/None → '—'."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return '—'
    if not np.isfinite(f):
        return '—'
    return f'{f:.{nd}f}{unit}'


def _corr_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """유한한 (x,y) 쌍에 대한 Pearson r, Spearman ρ, 표본수 반환."""
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, n
    xa, ya = x[m], y[m]
    if np.std(xa) == 0 or np.std(ya) == 0:
        return np.nan, np.nan, n
    pear = float(np.corrcoef(xa, ya)[0, 1])
    rx = pd.Series(xa).rank().values
    ry = pd.Series(ya).rank().values
    spear = float(np.corrcoef(rx, ry)[0, 1])
    return pear, spear, n


# ══════════════════════════════════════════════
#  Tab 8: 다중 트레이 트렌드뷰 + dOCV 산점도
# ══════════════════════════════════════════════

class Tab8Trend(QWidget):
    """
    여러 트레이의 SDM 값을 시각화.
    - 트렌드 모드: x축(트레이 > 셀 번호) 기준 SDM 산포
    - dOCV 산점도 모드: dOCV7 vs SDM(raw/보정후/z) — 대체재 검증
    - 점 클릭 → 해당 셀의 상세 정보 표시
    데이터: 현재 세션(메모리) + Tab7에서 내보낸 결과 파일(히스토리)
    """

    def __init__(self, state: AppState):
        super().__init__()
        self.state        = state
        self._hist_frames: list[pd.DataFrame] = []   # 추가 로드된 결과 파일
        # 클릭 조회용 플롯 상태
        self._plot_ax   = None
        self._plot_px   = None      # x 데이터 좌표
        self._plot_py   = None      # y 데이터 좌표
        self._plot_meta = None      # 각 점의 메타 DataFrame
        self._ann       = None      # 그래프 위 주석 객체
        self._build_ui()
        state.callbacks.append(self._refresh)

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # ── 왼쪽 컨트롤 (스크롤 가능) ─────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(260)
        cl = QVBoxLayout(ctrl)

        # 데이터 소스
        g_src = QGroupBox('데이터 소스')
        sl = QVBoxLayout(g_src)
        self.lbl_session = QLabel('현재 세션: (없음)')
        self.lbl_session.setWordWrap(True)
        sl.addWidget(self.lbl_session)
        self.btn_add = QPushButton('이전 결과 파일 추가 (Tab7 내보내기)…')
        self.btn_add.setToolTip('이전 결과 파일 추가 (Tab7 내보내기)')
        sl.addWidget(self.btn_add)
        self.lst_files = QListWidget()
        self.lst_files.setMinimumHeight(40)
        self.lst_files.setMaximumHeight(110)
        sl.addWidget(self.lst_files)
        self.btn_clear = QPushButton('추가 파일 초기화')
        sl.addWidget(self.btn_clear)
        cl.addWidget(g_src)

        # 표시 모드 + 트레이 선택
        g_mode = QGroupBox('표시')
        ml = QVBoxLayout(g_mode)
        ml.addWidget(QLabel('모드:'))
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(['트렌드 (트레이 > 셀)', 'dOCV 산점도'])
        ml.addWidget(self.cb_mode)
        ml.addWidget(QLabel('트레이:'))
        self.cb_tray = QComboBox()
        self.cb_tray.addItem('전체')
        ml.addWidget(self.cb_tray)
        cl.addWidget(g_mode)

        # Y축
        g_y = QGroupBox('Y축 (지표)')
        yl = QVBoxLayout(g_y)
        self.rb_raw  = QRadioButton('SDM raw (보정 전, µA)')
        self.rb_corr = QRadioButton('보정 후 (µA)')
        self.rb_z    = QRadioButton('z-score')
        self.rb_raw.setChecked(True)
        for rb in [self.rb_raw, self.rb_corr, self.rb_z]:
            yl.addWidget(rb)
        yl.addWidget(QLabel('분석 옵션 (보정/z-score 용):'))
        self.cb_opt = QComboBox()
        self.cb_opt.addItems([f'옵션 {i}' for i in range(1, 6)])
        yl.addWidget(self.cb_opt)
        cl.addWidget(g_y)

        # Y축 범위 / 스케일
        g_range = QGroupBox('Y축 범위 · 스케일')
        rl = QGridLayout(g_range)
        self.chk_yauto = QCheckBox('자동 범위')
        self.chk_yauto.setChecked(True)
        rl.addWidget(self.chk_yauto, 0, 0, 1, 2)
        rl.addWidget(QLabel('min'), 1, 0)
        self.spin_ymin = QDoubleSpinBox()
        self.spin_ymin.setRange(-1e6, 1e6); self.spin_ymin.setDecimals(3)
        self.spin_ymin.setValue(-1.0); self.spin_ymin.setEnabled(False)
        rl.addWidget(self.spin_ymin, 1, 1)
        rl.addWidget(QLabel('max'), 2, 0)
        self.spin_ymax = QDoubleSpinBox()
        self.spin_ymax.setRange(-1e6, 1e6); self.spin_ymax.setDecimals(3)
        self.spin_ymax.setValue(5.0); self.spin_ymax.setEnabled(False)
        rl.addWidget(self.spin_ymax, 2, 1)
        self.chk_log = QCheckBox('로그 스케일 (symlog)')
        self.chk_log.setToolTip('아웃라이어로 값이 0 부근에 압축될 때 유용. '
                                '음수·0도 표시되는 symlog 사용.')
        rl.addWidget(self.chk_log, 3, 0, 1, 2)
        cl.addWidget(g_range)

        # 필터
        g_filt = QGroupBox('필터')
        fl = QVBoxLayout(g_filt)
        fl.addWidget(QLabel('Rwiring 임계값 (0 = 미적용):'))
        self.spin_rw = QDoubleSpinBox()
        self.spin_rw.setRange(0, 100)
        self.spin_rw.setValue(0)
        self.spin_rw.setSuffix(' Ω')
        fl.addWidget(self.spin_rw)
        fl.addWidget(QLabel('히스토리 파일 N분 (i_*min 선택):'))
        self.spin_hist_n = QSpinBox()
        self.spin_hist_n.setRange(5, 30)
        self.spin_hist_n.setValue(15)
        self.spin_hist_n.setSuffix(' 분')
        fl.addWidget(self.spin_hist_n)
        cl.addWidget(g_filt)

        # 기준선
        g_bl = QGroupBox('기준선')
        bl = QVBoxLayout(g_bl)
        self.chk_baseline = QCheckBox('기준선 표시 (y)')
        bl.addWidget(self.chk_baseline)
        self.spin_baseline = QDoubleSpinBox()
        self.spin_baseline.setRange(-1000, 100000)
        self.spin_baseline.setValue(2.0)
        self.spin_baseline.setSingleStep(0.5)
        bl.addWidget(self.spin_baseline)
        cl.addWidget(g_bl)

        # 그리기
        self.btn_draw = QPushButton('▶  그래프 그리기')
        self.btn_draw.setFixedHeight(34)
        font = self.btn_draw.font(); font.setBold(True)
        self.btn_draw.setFont(font)
        cl.addWidget(self.btn_draw)

        self.lbl_stat = QLabel('')
        self.lbl_stat.setWordWrap(True)
        cl.addWidget(self.lbl_stat)

        # 선택 셀 정보
        g_det = QGroupBox('선택 셀 정보 (점 클릭)')
        dl = QVBoxLayout(g_det)
        self.lbl_detail = QLabel('그래프의 점을 클릭하세요.')
        self.lbl_detail.setWordWrap(True)
        self.lbl_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_detail.setStyleSheet('font-family:Consolas,monospace; font-size:11px;')
        dl.addWidget(self.lbl_detail)
        cl.addWidget(g_det)

        cl.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)
        scroll.setWidget(ctrl)
        layout.addWidget(scroll)

        # ── 오른쪽 캔버스 (+ 확대 툴바) ────────────
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        self.canvas = PlotCanvas(figsize=(13, 5))
        rlay.addWidget(NavigationToolbar2QT(self.canvas, self))
        rlay.addWidget(self.canvas)
        layout.addWidget(right)

        # 시그널
        self.btn_add.clicked.connect(self._add_files)
        self.btn_clear.clicked.connect(self._clear_files)
        self.btn_draw.clicked.connect(lambda: self._draw(show_empty_msg=True))
        self.chk_yauto.toggled.connect(self._on_yauto_toggled)
        # 뷰를 크게 바꾸는 컨트롤은 자동 재그리기
        for w in (self.cb_mode, self.cb_tray, self.cb_opt):
            w.currentIndexChanged.connect(lambda *_: self._draw())
        for rb in (self.rb_raw, self.rb_corr, self.rb_z):
            rb.toggled.connect(lambda *_: self._draw())
        self.chk_log.toggled.connect(lambda *_: self._draw())
        self.chk_baseline.toggled.connect(lambda *_: self._draw())
        self.spin_ymin.editingFinished.connect(self._draw)
        self.spin_ymax.editingFinished.connect(self._draw)
        # 점 클릭 조회
        self.canvas.mpl_connect('button_press_event', self._on_click)

    # ── 콜백 ──────────────────────────────────
    def _refresh(self):
        if not self.state.df_meta.empty:
            trays = self.state.df_meta['tray_id'].unique().tolist()
            preview = ', '.join(trays[:3]) + ('…' if len(trays) > 3 else '')
            self.lbl_session.setText(f'현재 세션: {len(trays)}개 트레이\n({preview})')
        else:
            self.lbl_session.setText('현재 세션: (없음)')
        self._refresh_tray_combo()

    def _refresh_tray_combo(self):
        """트레이 드롭다운을 현재 데이터로 갱신 (선택 유지)."""
        prev = self.cb_tray.currentText()
        trays = []
        if not self.state.df_meta.empty:
            trays = [str(t) for t in self.state.df_meta['tray_id'].unique().tolist()]
        for hdf in self._hist_frames:
            if 'tray_id' in hdf.columns:
                trays += [str(t) for t in hdf['tray_id'].dropna().unique().tolist()]
        # 중복 제거(순서 유지)
        seen, uniq = set(), []
        for t in trays:
            if t not in seen:
                seen.add(t); uniq.append(t)
        self.cb_tray.blockSignals(True)
        self.cb_tray.clear()
        self.cb_tray.addItem('전체')
        self.cb_tray.addItems(uniq)
        i = self.cb_tray.findText(prev)
        self.cb_tray.setCurrentIndex(i if i >= 0 else 0)
        self.cb_tray.blockSignals(False)

    def _on_yauto_toggled(self, checked: bool):
        self.spin_ymin.setEnabled(not checked)
        self.spin_ymax.setEnabled(not checked)
        self._draw()

    # ── 파일 관리 ──────────────────────────────
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, '결과 파일 선택 (Tab7 내보내기)', '',
            'Excel/CSV (*.xlsx *.xls *.csv);;모든 파일 (*)'
        )
        for path in paths:
            try:
                p = Path(path)
                if p.suffix.lower() in ('.xlsx', '.xls'):
                    df = pd.read_excel(p, header=0)
                else:
                    df = pd.read_csv(p, header=0, sep=None, engine='python')
                self._hist_frames.append(df)
                self.lst_files.addItem(p.name)
            except Exception as e:
                QMessageBox.warning(self, '파일 오류', f'{Path(path).name}:\n{e}')
        self._refresh_tray_combo()

    def _clear_files(self):
        self._hist_frames.clear()
        self.lst_files.clear()
        self._refresh_tray_combo()

    # ── 데이터 통합 ────────────────────────────
    def _collect_df(self) -> pd.DataFrame:
        """현재 세션 + 히스토리 파일 → 통합 DataFrame (필요 컬럼만)"""
        opt = self.cb_opt.currentIndex() + 1
        frames: list[pd.DataFrame] = []

        # 현재 세션
        if not self.state.df_meta.empty:
            df = self.state.df_meta.copy()
            n  = self.state.n_minutes

            # raw SDM (A → µA)
            raw_col = f'i_{n}min'
            df['_raw'] = pd.to_numeric(df.get(raw_col, np.nan), errors='coerce') * 1e6
            # 온도·전압·dOCV
            df['_temp'] = pd.to_numeric(df.get('t_final', np.nan), errors='coerce')
            df['_volt'] = pd.to_numeric(df.get('v_init',  np.nan), errors='coerce')
            df['_docv'] = pd.to_numeric(df.get(PROCESS_COL_DOCV, np.nan), errors='coerce')

            # 분석 결과
            res = self.state.analysis_results.get(opt)
            df['_corr'] = np.nan
            df['_z']    = np.nan
            if res:
                corr = res.get('corrected')
                z    = res.get('z_scores')
                if corr is not None:
                    df.loc[corr.index, '_corr'] = corr.values
                if z is not None:
                    df.loc[z.index, '_z'] = z.values

            keep_cols = ['tray_id', 'cell_no', 'rwiring',
                         '_raw', '_corr', '_z', '_temp', '_volt', '_docv']
            if PROCESS_COL_GRADE in df.columns:
                keep_cols.insert(3, PROCESS_COL_GRADE)
            frames.append(df[keep_cols])

        # 히스토리 파일
        for hdf in self._hist_frames:
            hdf2 = hdf.copy()

            # raw SDM — spin_hist_n으로 명시적 선택, 없으면 첫 번째 i_*min 폴백
            hist_n  = self.spin_hist_n.value()
            raw_col = f'i_{hist_n}min'
            if raw_col not in hdf2.columns:
                i_cols  = [c for c in hdf2.columns if c.startswith('i_') and c.endswith('min')]
                raw_col = i_cols[0] if i_cols else None
            hdf2['_raw'] = pd.to_numeric(hdf2[raw_col], errors='coerce') * 1e6 if raw_col else np.nan

            # 보정값
            corr_col = f'보정값_opt{opt}' if f'보정값_opt{opt}' in hdf2.columns else \
                       ('보정값' if '보정값' in hdf2.columns else None)
            hdf2['_corr'] = pd.to_numeric(hdf2[corr_col], errors='coerce') if corr_col else np.nan

            # z-score
            z_col = f'z_score_opt{opt}' if f'z_score_opt{opt}' in hdf2.columns else \
                    ('z_score' if 'z_score' in hdf2.columns else None)
            hdf2['_z'] = pd.to_numeric(hdf2[z_col], errors='coerce') if z_col else np.nan

            # 온도·전압·dOCV (Tab7 내보내기는 df_valid 기반이라 대개 존재)
            hdf2['_temp'] = pd.to_numeric(hdf2['t_final'], errors='coerce') if 't_final' in hdf2.columns else np.nan
            hdf2['_volt'] = pd.to_numeric(hdf2['v_init'],  errors='coerce') if 'v_init'  in hdf2.columns else np.nan
            hdf2['_docv'] = pd.to_numeric(hdf2[PROCESS_COL_DOCV], errors='coerce') if PROCESS_COL_DOCV in hdf2.columns else np.nan

            need = ['tray_id', 'cell_no', 'rwiring', PROCESS_COL_GRADE,
                    '_raw', '_corr', '_z', '_temp', '_volt', '_docv']
            avail = {c: hdf2[c] if c in hdf2.columns else pd.Series(np.nan, index=hdf2.index)
                     for c in need}
            frames.append(pd.DataFrame(avail))

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        # 중복 컬럼 제거 (grade 컬럼이 두 번 들어간 경우)
        return result.loc[:, ~result.columns.duplicated()]

    # ── 공통 준비 ──────────────────────────────
    def _prepare(self, show_empty_msg: bool):
        """수집 + 필터 + Y축 컬럼 선택. (df, y_col, y_label) 또는 None."""
        df = self._collect_df()
        if df.empty:
            if show_empty_msg:
                QMessageBox.information(self, '알림',
                    '표시할 데이터가 없습니다.\n'
                    '① 데이터 불러오기 탭에서 데이터를 먼저 로드하거나\n'
                    '이전 결과 파일(Tab7 내보내기)을 추가하세요.')
            return None

        # Rwiring 필터
        rw = self.spin_rw.value()
        if rw > 0 and 'rwiring' in df.columns:
            rw_vals = pd.to_numeric(df['rwiring'], errors='coerce').fillna(0)
            df = df[rw_vals <= rw].copy()

        # 트레이 필터
        df['tray_id'] = df['tray_id'].astype(str)
        tsel = self.cb_tray.currentText()
        if tsel and tsel != '전체':
            df = df[df['tray_id'] == tsel].copy()
        if df.empty:
            if show_empty_msg:
                QMessageBox.information(self, '알림', '선택한 조건에 해당하는 셀이 없습니다.')
            return None

        # Y축 컬럼
        if self.rb_raw.isChecked():
            y_col, y_label = '_raw',  'SDM 전류 (µA, 보정 전)'
        elif self.rb_corr.isChecked():
            y_col, y_label = '_corr', f'보정 후 전류 (µA, 옵션 {self.cb_opt.currentIndex()+1})'
        else:
            y_col, y_label = '_z',    f'z-score (옵션 {self.cb_opt.currentIndex()+1})'
        return df, y_col, y_label

    def _grade_masks(self, df: pd.DataFrame):
        if PROCESS_COL_GRADE in df.columns:
            grades = df[PROCESS_COL_GRADE].astype(str).str.strip().str.upper()
        else:
            grades = pd.Series('?', index=df.index)
        mask_A = (grades == 'A').values
        mask_E = (grades == 'E').values
        mask_U = ~(mask_A | mask_E)
        return mask_A, mask_E, mask_U

    def _apply_yaxis(self, ax):
        if self.chk_log.isChecked():
            ax.set_yscale('symlog')
        if not self.chk_yauto.isChecked():
            lo, hi = self.spin_ymin.value(), self.spin_ymax.value()
            if hi > lo:
                ax.set_ylim(lo, hi)

    # ── 그래프 진입점 ──────────────────────────
    def _draw(self, show_empty_msg: bool = False):
        prep = self._prepare(show_empty_msg)
        if prep is None:
            return
        df, y_col, y_label = prep
        # 이전 클릭 주석 초기화
        self._ann = None
        if self.cb_mode.currentIndex() == 1:
            self._draw_scatter(df, y_col, y_label)
        else:
            self._draw_trend(df, y_col, y_label)

    # ── 트렌드 모드 ────────────────────────────
    def _draw_trend(self, df: pd.DataFrame, y_col: str, y_label: str):
        df = df.copy()
        df['cell_no'] = pd.to_numeric(df['cell_no'], errors='coerce')
        df = df.sort_values(['tray_id', 'cell_no']).reset_index(drop=True)

        x_pos  = np.arange(len(df), dtype=float)
        y_vals = pd.to_numeric(df[y_col], errors='coerce').values if y_col in df.columns \
                 else np.full(len(df), np.nan)
        mask_A, mask_E, mask_U = self._grade_masks(df)

        # 트레이 경계
        tray_arr   = df['tray_id'].values
        boundaries = [0] + [i for i in range(1, len(tray_arr)) if tray_arr[i] != tray_arr[i-1]] + [len(df)]
        tray_mids  = [(boundaries[i] + boundaries[i+1] - 1) / 2 for i in range(len(boundaries)-1)]
        tray_names = [tray_arr[boundaries[i]] for i in range(len(boundaries)-1)]

        n_trays   = len(tray_names)
        label_fs  = max(5, 8 - max(0, n_trays - 6))
        fig_width = max(12, n_trays * 1.8)
        self.canvas.fig.set_size_inches(fig_width, 5)
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)

        if mask_U.any():
            ax.scatter(x_pos[mask_U], y_vals[mask_U],
                       color='#888888', s=7, alpha=0.35, label='미분류', zorder=2)
        if mask_A.any():
            ax.scatter(x_pos[mask_A], y_vals[mask_A],
                       color='black',   s=7, alpha=0.45, label='양품(A)', zorder=3)
        if mask_E.any():
            ax.scatter(x_pos[mask_E], y_vals[mask_E],
                       color='red',     s=11, alpha=0.85, label='불량(E)', zorder=4)

        for b in boundaries[1:-1]:
            ax.axvline(b - 0.5, color='#bbbbbb', linestyle='--', linewidth=0.8, zorder=1)

        if self.chk_baseline.isChecked():
            blv = self.spin_baseline.value()
            ax.axhline(blv, color='royalblue', linestyle='--', linewidth=1.2,
                       label=f'기준선  y = {blv}', zorder=5)

        ax.set_xticks(tray_mids)
        ax.set_xticklabels(tray_names, rotation=45, ha='right', fontsize=label_fs)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title('다중 트레이 SDM 트렌드  (X축: 트레이 > 셀 번호 순)', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.15, axis='y')
        ax.set_xlim(-1, len(df))
        self._apply_yaxis(ax)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        # 클릭 조회용 상태 저장
        self._plot_ax   = ax
        self._plot_px   = x_pos
        self._plot_py   = y_vals
        self._plot_meta = df

        n_total = len(df); n_A = int(mask_A.sum()); n_E = int(mask_E.sum())
        rate = f'{n_E/n_total*100:.1f}%' if n_total > 0 else '—'
        self.lbl_stat.setText(f'총 {n_total}셀  |  양품 {n_A}  |  불량 {n_E}  |  불량률 {rate}')

    # ── dOCV 산점도 모드 ───────────────────────
    def _draw_scatter(self, df: pd.DataFrame, y_col: str, y_label: str):
        df = df.reset_index(drop=True)
        x = pd.to_numeric(df['_docv'], errors='coerce').values
        y = pd.to_numeric(df[y_col], errors='coerce').values if y_col in df.columns \
            else np.full(len(df), np.nan)
        mask_A, mask_E, mask_U = self._grade_masks(df)

        self.canvas.fig.set_size_inches(9, 6)
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)

        if mask_U.any():
            ax.scatter(x[mask_U], y[mask_U], color='#888888', s=12, alpha=0.4, label='미분류', zorder=2)
        if mask_A.any():
            ax.scatter(x[mask_A], y[mask_A], color='black', s=12, alpha=0.5, label='양품(A)', zorder=3)
        if mask_E.any():
            ax.scatter(x[mask_E], y[mask_E], color='red', s=22, alpha=0.85, label='불량(E)', zorder=4)

        # 전체 상관 + 회귀선
        r_all, rho_all, n_all = _corr_pair(x, y)
        fin = np.isfinite(x) & np.isfinite(y)
        if fin.sum() >= 2:
            slope, intercept = np.polyfit(x[fin], y[fin], 1)
            xs = np.linspace(np.nanmin(x[fin]), np.nanmax(x[fin]), 100)
            ax.plot(xs, slope * xs + intercept, color='royalblue',
                    linestyle='--', linewidth=1.2, label='회귀선(전체)', zorder=5)

        # 양품만(E 제외) 상관 — 대체재 vs 단순 단락검출기 판정
        good = ~mask_E
        r_good, rho_good, n_good = _corr_pair(x[good], y[good])

        if self.chk_baseline.isChecked():
            blv = self.spin_baseline.value()
            ax.axhline(blv, color='seagreen', linestyle=':', linewidth=1.1,
                       label=f'기준선  y = {blv}', zorder=5)

        txt = (f'전체 (n={n_all}):  r={_fmt(r_all,nd=3)}  ρ={_fmt(rho_all,nd=3)}\n'
               f'E제외 (n={n_good}):  r={_fmt(r_good,nd=3)}  ρ={_fmt(rho_good,nd=3)}')
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
                fontsize=9, bbox=dict(boxstyle='round', fc='#f5f5f5', ec='#cccccc', alpha=0.9))

        tsel = self.cb_tray.currentText()
        ax.set_xlabel('dOCV7  (Delta OCV #07, mV)', fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(f'dOCV7 vs SDM  —  {tsel}', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.15)
        self._apply_yaxis(ax)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        # 클릭 조회용 상태 저장
        self._plot_ax   = ax
        self._plot_px   = x.astype(float)
        self._plot_py   = y.astype(float)
        self._plot_meta = df

        interp = self._surrogate_verdict(r_good, n_good)
        self.lbl_stat.setText(
            f'전체 r={_fmt(r_all,nd=3)} / ρ={_fmt(rho_all,nd=3)}  ·  '
            f'E제외 r={_fmt(r_good,nd=3)} / ρ={_fmt(rho_good,nd=3)}\n{interp}'
        )

    @staticmethod
    def _surrogate_verdict(r_good: float, n_good: int) -> str:
        if not np.isfinite(r_good) or n_good < 10:
            return '판정: 표본 부족 — 상관 신뢰 어려움'
        a = abs(r_good)
        if a >= 0.5:
            return '판정: 양품 구간에서도 dOCV를 따라감 → 대체재 가능성'
        if a >= 0.2:
            return '판정: 약한 상관 — 경계성 불량 재검증 필요'
        return '판정: 양품 구간 무상관 → 단순 단락검출기에 가까움'

    # ── 점 클릭 조회 ───────────────────────────
    def _on_click(self, event):
        if event.inaxes is None or self._plot_meta is None:
            return
        if event.x is None or self._plot_ax is None:
            return
        px = np.asarray(self._plot_px, dtype=float)
        py = np.asarray(self._plot_py, dtype=float)
        finite = np.isfinite(px) & np.isfinite(py)
        if not finite.any():
            return
        pts = np.column_stack([px[finite], py[finite]])
        disp = self._plot_ax.transData.transform(pts)
        d = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
        j = int(np.argmin(d))
        if d[j] > 30:      # 픽셀 임계
            return
        idx = int(np.nonzero(finite)[0][j])
        self._show_detail(idx, px[idx], py[idx])

    def _show_detail(self, idx: int, px: float, py: float):
        m = self._plot_meta.iloc[idx]
        try:
            cell = str(int(float(m['cell_no'])))
        except (TypeError, ValueError):
            cell = '—'
        tray = str(m.get('tray_id', '—'))
        self.lbl_detail.setText(
            f'트레이  {tray}\n'
            f'셀 번호  {cell}\n'
            f'──────────────\n'
            f'Rwiring  {_fmt(m.get("rwiring"), " Ω")}\n'
            f'SDM raw  {_fmt(m.get("_raw"), " µA")}\n'
            f'보정값   {_fmt(m.get("_corr"), " µA")}\n'
            f'z-score  {_fmt(m.get("_z"))}\n'
            f'온도     {_fmt(m.get("_temp"), " °C", 2)}\n'
            f'전압     {_fmt(m.get("_volt"), " mV", 1)}\n'
            f'dOCV7    {_fmt(m.get("_docv"), " mV")}'
        )
        # 그래프 위 주석
        if self._ann is not None:
            try:
                self._ann.remove()
            except Exception:
                pass
        self._ann = self._plot_ax.annotate(
            f'{tray}·#{cell}',
            xy=(px, py), xytext=(12, 12), textcoords='offset points',
            fontsize=8, color='darkblue',
            bbox=dict(boxstyle='round', fc='yellow', ec='darkblue', alpha=0.85),
            arrowprops=dict(arrowstyle='->', color='darkblue'),
            zorder=10,
        )
        self.canvas.draw_idle()
