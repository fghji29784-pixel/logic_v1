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




# ══════════════════════════════════════════════
#  Tab 8: 다중 트레이 트렌드뷰
# ══════════════════════════════════════════════

class Tab8Trend(QWidget):
    """
    여러 트레이의 SDM 값을 x축(트레이>채널) 기준으로 시각화.
    - 같은 세션: 현재 메모리 데이터 자동 사용
    - 히스토리: Tab7에서 내보낸 Excel/CSV 파일 추가 로드
    """

    def __init__(self, state: AppState):
        super().__init__()
        self.state        = state
        self._hist_frames: list[pd.DataFrame] = []   # 추가 로드된 결과 파일
        self._build_ui()
        state.callbacks.append(self._refresh)

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # ── 왼쪽 컨트롤 ──────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(250)
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
        self.lst_files.setMinimumHeight(50)
        self.lst_files.setMaximumHeight(150)
        sl.addWidget(self.lst_files)
        self.btn_clear = QPushButton('추가 파일 초기화')
        sl.addWidget(self.btn_clear)
        cl.addWidget(g_src)

        # Y축
        g_y = QGroupBox('Y축')
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
        self.chk_baseline = QCheckBox('기준선 표시')
        bl.addWidget(self.chk_baseline)
        self.spin_baseline = QDoubleSpinBox()
        self.spin_baseline.setRange(-1000, 100000)
        self.spin_baseline.setValue(2.0)
        self.spin_baseline.setSingleStep(0.5)
        bl.addWidget(self.spin_baseline)
        cl.addWidget(g_bl)

        # 그리기
        self.btn_draw = QPushButton('▶  그래프 그리기')
        self.btn_draw.setFixedHeight(36)
        font = self.btn_draw.font(); font.setBold(True)
        self.btn_draw.setFont(font)
        cl.addWidget(self.btn_draw)

        self.lbl_stat = QLabel('')
        self.lbl_stat.setWordWrap(True)
        cl.addWidget(self.lbl_stat)
        cl.addStretch()
        layout.addWidget(ctrl)

        # ── 오른쪽 캔버스 ─────────────────────────
        self.canvas = PlotCanvas(figsize=(13, 5))
        layout.addWidget(self.canvas)

        # 시그널
        self.btn_add.clicked.connect(self._add_files)
        self.btn_clear.clicked.connect(self._clear_files)
        self.btn_draw.clicked.connect(self._draw)

    # ── 콜백 ──────────────────────────────────
    def _refresh(self):
        if not self.state.df_meta.empty:
            trays = self.state.df_meta['tray_id'].unique().tolist()
            preview = ', '.join(trays[:3]) + ('…' if len(trays) > 3 else '')
            self.lbl_session.setText(f'현재 세션: {len(trays)}개 트레이\n({preview})')
        else:
            self.lbl_session.setText('현재 세션: (없음)')

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

    def _clear_files(self):
        self._hist_frames.clear()
        self.lst_files.clear()

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

            keep_cols = ['tray_id', 'cell_no', 'rwiring', '_raw', '_corr', '_z']
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

            need = ['tray_id', 'cell_no', 'rwiring', PROCESS_COL_GRADE, '_raw', '_corr', '_z']
            avail = {c: hdf2[c] if c in hdf2.columns else pd.Series(np.nan, index=hdf2.index)
                     for c in need}
            frames.append(pd.DataFrame(avail))

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        # 중복 컬럼 제거 (grade 컬럼이 두 번 들어간 경우)
        return result.loc[:, ~result.columns.duplicated()]

    # ── 그래프 ────────────────────────────────
    def _draw(self):
        df = self._collect_df()
        if df.empty:
            QMessageBox.information(self, '알림',
                '표시할 데이터가 없습니다.\n'
                '① 데이터 불러오기 탭에서 데이터를 먼저 로드하거나\n'
                '이전 결과 파일(Tab7 내보내기)을 추가하세요.')
            return

        # Rwiring 필터
        rw = self.spin_rw.value()
        if rw > 0 and 'rwiring' in df.columns:
            rw_vals = pd.to_numeric(df['rwiring'], errors='coerce').fillna(0)
            df = df[rw_vals <= rw].copy()

        # Y축 컬럼
        if self.rb_raw.isChecked():
            y_col, y_label = '_raw',  'SDM 전류 (µA, 보정 전)'
        elif self.rb_corr.isChecked():
            y_col, y_label = '_corr', f'보정 후 전류 (µA, 옵션 {self.cb_opt.currentIndex()+1})'
        else:
            y_col, y_label = '_z',    f'z-score (옵션 {self.cb_opt.currentIndex()+1})'

        # 정렬
        df['tray_id'] = df['tray_id'].astype(str)
        df['cell_no'] = pd.to_numeric(df['cell_no'], errors='coerce')
        df = df.sort_values(['tray_id', 'cell_no']).reset_index(drop=True)

        x_pos  = np.arange(len(df))
        y_vals = pd.to_numeric(df[y_col], errors='coerce').values if y_col in df.columns \
                 else np.full(len(df), np.nan)

        # 등급 마스크
        grade_col = PROCESS_COL_GRADE
        if grade_col in df.columns:
            grades = df[grade_col].astype(str).str.strip().str.upper()
        else:
            grades = pd.Series('?', index=df.index)
        mask_A = (grades == 'A').values
        mask_E = (grades == 'E').values
        mask_U = ~(mask_A | mask_E)

        # 트레이 경계 계산
        tray_arr   = df['tray_id'].values
        boundaries = [0] + [i for i in range(1, len(tray_arr)) if tray_arr[i] != tray_arr[i-1]] + [len(df)]
        tray_mids  = [(boundaries[i] + boundaries[i+1] - 1) / 2 for i in range(len(boundaries)-1)]
        tray_names = [tray_arr[boundaries[i]] for i in range(len(boundaries)-1)]

        # ── 그리기 ──
        n_trays   = len(tray_names)
        label_fs  = max(5, 8 - max(0, n_trays - 6))   # 트레이 많을수록 폰트 축소
        fig_width = max(12, n_trays * 1.8)             # 트레이 수에 비례한 가로 폭
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

        # 트레이 구분 수직선
        for b in boundaries[1:-1]:
            ax.axvline(b - 0.5, color='#bbbbbb', linestyle='--', linewidth=0.8, zorder=1)

        # 기준선
        if self.chk_baseline.isChecked():
            bl = self.spin_baseline.value()
            ax.axhline(bl, color='royalblue', linestyle='--', linewidth=1.2,
                       label=f'기준선  y = {bl}', zorder=5)

        # 축 설정
        ax.set_xticks(tray_mids)
        ax.set_xticklabels(tray_names, rotation=45, ha='right', fontsize=label_fs)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title('다중 트레이 SDM 트렌드  (X축: 트레이 > 셀 번호 순)', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.15, axis='y')
        ax.set_xlim(-1, len(df))

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        # 통계
        n_total = len(df)
        n_A = int(mask_A.sum())
        n_E = int(mask_E.sum())
        rate = f'{n_E/n_total*100:.1f}%' if n_total > 0 else '—'
        self.lbl_stat.setText(
            f'총 {n_total}셀  |  양품 {n_A}  |  불량 {n_E}  |  불량률 {rate}'
        )
