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
#  Tab 3: 히트맵
# ══════════════════════════════════════════════

class Tab3Heatmap(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._build_ui()
        state.callbacks.append(self._refresh)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('트레이:'))
        self.cb_tray = QComboBox()
        ctrl.addWidget(self.cb_tray)

        ctrl.addWidget(QLabel('히트맵 종류:'))
        self.cb_type = QComboBox()
        self.cb_type.addItems([
            'SDM 전류 (보정전/후+dOCV+Rwiring)',
            '온도 (T_start/T_final/ΔT)',
            '전압 (V_init/V_final/ΔV)',
        ])
        ctrl.addWidget(self.cb_type)

        self.chk_clip = QCheckBox('클리핑')
        self.chk_clip.setChecked(True)
        self.chk_clip.setToolTip(
            '클리핑: 상위 N% 를 넘는 극단값을 색상 상한으로 눌러주는 표시 기능.\n'
            '불량셀 1~2개의 큰 값 때문에 나머지 정상 셀이 전부 같은 색으로\n'
            '뭉개져 안 보이는 것을 방지해 색 대비를 확보합니다.\n'
            '(데이터 값 자체는 바뀌지 않고, 색 스케일만 조정)')
        ctrl.addWidget(self.chk_clip)
        self.spin_clip = QSpinBox()
        self.spin_clip.setRange(80, 100)
        self.spin_clip.setValue(99)
        self.spin_clip.setSuffix('%')
        self.spin_clip.setFixedWidth(60)
        self.spin_clip.setToolTip('색상 상한으로 쓸 백분위. 예: 99% → 상위 1% 극단값을 상한색으로 표시.')
        ctrl.addWidget(self.spin_clip)

        ctrl.addWidget(QLabel('색상:'))
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(['빨강-초록 (RdYlGn)', '색약안전 (cividis)', '색약안전 (viridis)'])
        self.cb_cmap.setToolTip('빨강-초록은 적록색약에 취약합니다. 색약안전 팔레트를 권장합니다.\n'
                                '(불량 E셀은 색과 무관하게 빨간 테두리로도 표시됩니다.)')
        ctrl.addWidget(self.cb_cmap)

        self.btn_draw = QPushButton('그리기')
        ctrl.addWidget(self.btn_draw)
        self.btn_save_all = QPushButton('전체 저장 (PNG)')
        ctrl.addWidget(self.btn_save_all)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        _clip_help = QLabel(
            '※ 클리핑: 극단값(상위 N% 초과)을 색상 상한으로 제한해 색 대비를 높이는 기능입니다. '
            '값 자체는 유지되며 색 스케일만 조정됩니다.  |  빨간 테두리 = E등급(불량) 셀  |  '
            '아래 툴바(🔍)로 확대해 셀 값을 크게 볼 수 있습니다.')
        _clip_help.setWordWrap(True)
        _clip_help.setStyleSheet('color:#777; font-size:10px;')
        layout.addWidget(_clip_help)

        self.canvas = PlotCanvas(figsize=(13, 5.5))
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

        self.btn_draw.clicked.connect(self._draw)
        self.btn_save_all.clicked.connect(self._save_all)
        self.cb_cmap.currentIndexChanged.connect(self._draw)

    def _refresh(self):
        self.cb_tray.clear()
        if self.state.df_meta.empty:
            return
        trays = self.state.df_meta['tray_id'].unique().tolist()
        self.cb_tray.addItems(trays)

    def _get_grid(self, tray: str, col: str) -> np.ndarray:
        """셀 번호 → 12×12 그리드 변환 (y=1~12, x=A~L)"""
        sub = self.state.df_meta[self.state.df_meta['tray_id'] == tray]
        grid = np.full((TRAY_ROWS, TRAY_COLS), np.nan)
        if col not in sub.columns:
            return grid
        for _, row in sub.iterrows():
            cn    = int(row['cell_no']) - 1
            x_idx = cn // TRAY_COLS   # A~L (0~11)
            y_idx = cn %  TRAY_COLS   # 1~12 (0~11)
            try:
                grid[y_idx, x_idx] = float(row[col])
            except (ValueError, TypeError):
                pass
        return grid

    def _cmap(self) -> str:
        """선택된 컬러맵 (색약안전 옵션 포함). 높은 값 = 눈에 띄는 색."""
        return {0: 'RdYlGn_r', 1: 'cividis', 2: 'viridis'}.get(
            self.cb_cmap.currentIndex(), 'RdYlGn_r')

    def _draw(self):
        tray = self.cb_tray.currentText()
        kind = self.cb_type.currentIndex()
        if not tray or self.state.df_meta.empty:
            return
        self._draw_core(self.canvas.fig, tray, kind)
        self.canvas.draw()

    def _clip_note(self) -> str:
        return (f'  (상위 {self.spin_clip.value()}% 클리핑)'
                if self.chk_clip.isChecked() else '')

    def _set_ticks(self, ax):
        row_labels = [chr(ord('A') + i) for i in range(TRAY_COLS)]
        ax.set_xticks(range(TRAY_COLS))
        ax.set_yticks(range(TRAY_ROWS))
        ax.set_xticklabels(row_labels, fontsize=6)
        ax.set_yticklabels(range(1, TRAY_ROWS + 1), fontsize=6)
        ax.set_xlabel('행 (A–L)', fontsize=7)
        ax.set_ylabel('열 (1–12)', fontsize=7)

    def _render_heat(self, fig, ax, grid, title, fmt, e_cells=None,
                     fontsize=7, vmin=None, vmax=None):
        """연속값 히트맵 한 패널 렌더 (클리핑·셀값 주석·colorbar·E테두리 포함).
        vmin/vmax 를 주면 색축 고정(패널 간 동일 스케일)."""
        if vmax is None and self.chk_clip.isChecked():
            flat = grid[~np.isnan(grid)]
            if len(flat) > 0:
                vmax = np.percentile(flat, self.spin_clip.value())
        im = ax.imshow(grid, cmap=self._cmap(), aspect='equal', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        norm_obj = im.norm
        cmap_obj = im.cmap
        for iy in range(TRAY_ROWS):
            for ix in range(TRAY_COLS):
                val = grid[iy, ix]
                if not np.isnan(val):
                    rgba = cmap_obj(norm_obj(val))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    tc = 'white' if lum < 0.5 else 'black'
                    ax.text(ix, iy, fmt.format(val),
                            ha='center', va='center', fontsize=fontsize, color=tc)
        # E등급(불량) 셀 빨간 테두리 강조
        if e_cells:
            for (iy, ix) in e_cells:
                ax.add_patch(Rectangle((ix - 0.5, iy - 0.5), 1, 1, fill=False,
                                       edgecolor='red', lw=2.0, zorder=5))
        self._set_ticks(ax)

    def _shared_range(self, *grids):
        """여러 그리드의 공통 색축(vmin, vmax) 반환 (클리핑 반영)."""
        parts = [g[~np.isnan(g)] for g in grids if g is not None]
        vals  = np.concatenate(parts) if parts else np.array([])
        if len(vals) == 0:
            return None, None
        vmin = float(vals.min())
        vmax = (float(np.percentile(vals, self.spin_clip.value()))
                if self.chk_clip.isChecked() else float(vals.max()))
        return vmin, vmax

    def _e_cells(self, tray: str) -> set:
        """해당 트레이 E등급 셀의 (iy, ix) 격자 좌표 집합."""
        cells = set()
        df = self.state.df_meta
        if PROCESS_COL_GRADE not in df.columns:
            return cells
        sub = df[df['tray_id'] == tray]
        for _, row in sub.iterrows():
            if str(row.get(PROCESS_COL_GRADE, '')).strip().upper() == 'E':
                try:
                    cn = int(row['cell_no']) - 1
                except (ValueError, TypeError):
                    continue
                cells.add((cn % TRAY_COLS, cn // TRAY_COLS))
        return cells

    def _get_corrected_grid(self, tray: str) -> np.ndarray:
        """선택 옵션의 SDM 보정값을 12×12 그리드로 (트레이별 회귀 결과 우선)."""
        grid    = np.full((TRAY_ROWS, TRAY_COLS), np.nan)
        results = self.state.analysis_results
        if not results:
            return grid
        res = results.get(self.state.selected_option) or results[sorted(results)[0]]
        per_tray = res.get('per_tray_results', {})
        if tray in per_tray:
            dfv  = per_tray[tray].get('df_valid')
            corr = per_tray[tray].get('corrected')
        else:
            dfv  = res.get('df_valid')
            corr = res.get('corrected')
        if dfv is None or corr is None or dfv.empty:
            return grid
        dfv = dfv.copy()
        dfv['_corr'] = corr
        if 'tray_id' in dfv.columns:
            dfv = dfv[dfv['tray_id'].astype(str) == str(tray)]
        for _, row in dfv.iterrows():
            try:
                cn = int(row['cell_no']) - 1
            except (ValueError, TypeError):
                continue
            v = row['_corr']
            if pd.notna(v):
                grid[cn % TRAY_COLS, cn // TRAY_COLS] = float(v)
        return grid

    def _draw_core(self, fig, tray: str, kind: int):
        n  = self.state.n_minutes
        ec = self._e_cells(tray)   # E등급 셀 (모든 패널에 빨간 테두리)
        fig.clear()

        if kind == 0:
            # SDM 전류: 보정 전 / 보정 후 / dOCV / Rwiring 한 번에
            axes = fig.subplots(1, 4)
            self._render_heat(fig, axes[0],
                              self._get_grid(tray, f'i_{n}min') * 1e6,
                              'SDM 보정 전 (µA)', '{:.1f}', ec, fontsize=5)
            self._render_heat(fig, axes[1], self._get_corrected_grid(tray),
                              f'SDM 보정 후 (µA, 옵션{self.state.selected_option})',
                              '{:.1f}', ec, fontsize=5)
            self._render_heat(fig, axes[2],
                              self._get_grid(tray, PROCESS_COL_DOCV),
                              'dOCV #07', '{:.1f}', ec, fontsize=5)
            self._render_heat(fig, axes[3], self._get_grid(tray, 'rwiring'),
                              'Rwiring (Ω)', '{:.3f}', ec, fontsize=5)
            fig.suptitle(f'{tray} — SDM 전류{self._clip_note()}  '
                         f'(빨간 테두리=E등급)', fontsize=10)

        elif kind == 1:
            # 온도: T_start / T_final / ΔT — start·final 색축 동일
            axes = fig.subplots(1, 3)
            g_s = self._get_grid(tray, 't_init')
            g_f = self._get_grid(tray, 't_final')
            vmin, vmax = self._shared_range(g_s, g_f)
            self._render_heat(fig, axes[0], g_s, 'T_start (°C)', '{:.2f}',
                              ec, fontsize=7, vmin=vmin, vmax=vmax)
            self._render_heat(fig, axes[1], g_f, 'T_final (°C)', '{:.2f}',
                              ec, fontsize=7, vmin=vmin, vmax=vmax)
            self._render_heat(fig, axes[2], g_s - g_f,
                              'ΔT = T_start−T_final (°C)', '{:.2f}', ec, fontsize=7)
            fig.suptitle(f'{tray} — 온도  (T_start·T_final 색축 동일, '
                         f'빨간 테두리=E등급)', fontsize=10)

        elif kind == 2:
            # 전압: V_init / V_final / ΔV — init·final 색축 동일
            axes = fig.subplots(1, 3)
            g_i = self._get_grid(tray, 'v_init')
            g_f = self._get_grid(tray, 'v_final')
            vmin, vmax = self._shared_range(g_i, g_f)
            self._render_heat(fig, axes[0], g_i, 'V_init (mV)', '{:.1f}',
                              ec, fontsize=7, vmin=vmin, vmax=vmax)
            self._render_heat(fig, axes[1], g_f, 'V_final (mV)', '{:.1f}',
                              ec, fontsize=7, vmin=vmin, vmax=vmax)
            self._render_heat(fig, axes[2], g_i - g_f,
                              'ΔV = V_init−V_final (mV)', '{:.2f}', ec, fontsize=7)
            fig.suptitle(f'{tray} — 전압  (V_init·V_final 색축 동일, '
                         f'빨간 테두리=E등급)', fontsize=10)

        fig.tight_layout()

    def _save_all(self):
        if self.state.df_meta.empty:
            QMessageBox.warning(self, '경고', '데이터를 먼저 불러오세요.')
            return
        folder = QFileDialog.getExistingDirectory(self, '전체 히트맵 저장 폴더 선택')
        if not folder:
            return
        import matplotlib.pyplot as plt
        trays = self.state.df_meta['tray_id'].unique().tolist()
        kind_names = ['SDM전류', '온도', '전압']
        count = 0
        for tray in trays:
            for kind, name in enumerate(kind_names):
                fig = plt.figure(figsize=(18, 5) if kind == 0 else (14, 5))
                self._draw_core(fig, tray, kind)
                fig.savefig(f'{folder}/{tray}_{name}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                count += 1
        QMessageBox.information(self, '완료', f'{count}개 히트맵 저장 완료\n{folder}')
