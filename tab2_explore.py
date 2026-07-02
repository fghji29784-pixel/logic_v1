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
#  Tab 2: 원시 데이터 탐색
# ══════════════════════════════════════════════

class Tab2Explore(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._highlighted_cell: int | None = None
        self._line_refs: dict = {}
        self._build_ui()
        state.callbacks.append(self._refresh)

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # ── 왼쪽 컨트롤 ──────────────────────────
        ctrl = QWidget()
        ctrl.setFixedWidth(230)
        cl = QVBoxLayout(ctrl)

        cl.addWidget(QLabel('트레이:'))
        self.cb_tray = QComboBox()
        cl.addWidget(self.cb_tray)

        # 오버레이 그룹
        g_ov = QGroupBox('전체 오버레이')
        ovl = QVBoxLayout(g_ov)

        h_btns = QHBoxLayout()
        self.btn_all  = QPushButton('전체 선택')
        self.btn_none = QPushButton('전체 해제')
        self.btn_all.setFixedHeight(24)
        self.btn_none.setFixedHeight(24)
        h_btns.addWidget(self.btn_all)
        h_btns.addWidget(self.btn_none)
        ovl.addLayout(h_btns)

        ovl.addWidget(QLabel('제외 셀 (체크 해제):'))
        self.lst_cells = QListWidget()
        self.lst_cells.setMinimumHeight(120)
        self.lst_cells.setMaximumHeight(220)
        ovl.addWidget(self.lst_cells)

        self.btn_draw_ov = QPushButton('▶ 오버레이 그리기')
        self.btn_draw_ov.setFixedHeight(30)
        font_b = self.btn_draw_ov.font()
        font_b.setBold(True)
        self.btn_draw_ov.setFont(font_b)
        ovl.addWidget(self.btn_draw_ov)

        self.lbl_picked = QLabel('선택: —')
        self.lbl_picked.setWordWrap(True)
        font_p = self.lbl_picked.font()
        font_p.setBold(True)
        self.lbl_picked.setFont(font_p)
        self.lbl_picked.setStyleSheet(
            'color: royalblue; background: #eef4ff;'
            ' padding: 4px; border-radius: 3px;'
        )
        ovl.addWidget(self.lbl_picked)
        cl.addWidget(g_ov)

        # 단일 셀 그룹
        g_sc = QGroupBox('단일 셀 보기')
        scl = QVBoxLayout(g_sc)
        scl.addWidget(QLabel('셀 번호:'))
        self.cb_cell = QComboBox()
        scl.addWidget(self.cb_cell)
        self.btn_plot = QPushButton('그래프 그리기')
        scl.addWidget(self.btn_plot)
        cl.addWidget(g_sc)

        cl.addStretch()
        layout.addWidget(ctrl)

        # ── 오른쪽: 툴바 + 캔버스 ────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.canvas = PlotCanvas(figsize=(10, 7))
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        rl.addWidget(self.toolbar)
        rl.addWidget(self.canvas)
        layout.addWidget(right)

        # 시그널
        self.cb_tray.currentTextChanged.connect(self._update_cells)
        self.btn_all.clicked.connect(self._select_all)
        self.btn_none.clicked.connect(self._deselect_all)
        self.btn_draw_ov.clicked.connect(self._plot_overlay)
        self.btn_plot.clicked.connect(self._plot_cell)
        self.canvas.mpl_connect('pick_event', self._on_pick)

    def _refresh(self):
        self.cb_tray.clear()
        if self.state.df_ts.empty:
            return
        trays = self.state.df_ts['tray_id'].unique().tolist()
        self.cb_tray.addItems(trays)

    def _update_cells(self, tray: str):
        self.cb_cell.clear()
        self.lst_cells.clear()
        if self.state.df_ts.empty or not tray:
            return

        cells = sorted(
            self.state.df_ts[self.state.df_ts['tray_id'] == tray]['cell_no']
            .dropna().astype(int).unique()
        )
        self.cb_cell.addItems([str(c) for c in cells])

        grade_col  = PROCESS_COL_GRADE
        grade_map: dict = {}
        if not self.state.df_meta.empty and grade_col in self.state.df_meta.columns:
            sub = self.state.df_meta[
                self.state.df_meta['tray_id'] == tray
            ].dropna(subset=['cell_no'])
            grade_map = dict(zip(
                sub['cell_no'].astype(int),
                sub[grade_col].astype(str).str.strip().str.upper(),
            ))

        for c in cells:
            grade = grade_map.get(c, '')
            text  = f'{c}  ({cell_to_label(c)})'
            if grade == 'E':
                text += '  ★E'
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, int(c))
            item.setCheckState(Qt.Checked)
            if grade == 'E':
                item.setForeground(QColor(200, 0, 0))
            self.lst_cells.addItem(item)

        QTimer.singleShot(50, self._plot_overlay)

    def _select_all(self):
        for i in range(self.lst_cells.count()):
            self.lst_cells.item(i).setCheckState(Qt.Checked)

    def _deselect_all(self):
        for i in range(self.lst_cells.count()):
            self.lst_cells.item(i).setCheckState(Qt.Unchecked)

    def _get_checked_cells(self) -> set:
        checked = set()
        for i in range(self.lst_cells.count()):
            item = self.lst_cells.item(i)
            if item.checkState() == Qt.Checked:
                checked.add(int(item.data(Qt.UserRole)))
        return checked

    def _get_cell_color(self, tray: str, cell_no: int) -> tuple:
        grade_col = PROCESS_COL_GRADE
        if not self.state.df_meta.empty and grade_col in self.state.df_meta.columns:
            sub = self.state.df_meta[
                (self.state.df_meta['tray_id'] == tray) &
                (self.state.df_meta['cell_no'].astype(int) == cell_no)
            ]
            if not sub.empty:
                grade = str(sub.iloc[0][grade_col]).strip().upper()
                if grade == 'E':
                    return 'red', 0.75
        return 'black', 0.30

    def _plot_overlay(self):
        tray = self.cb_tray.currentText()
        if not tray or self.state.df_ts.empty:
            return

        self._line_refs        = {}
        self._highlighted_cell = None
        checked = self._get_checked_cells()
        sub     = self.state.df_ts[self.state.df_ts['tray_id'] == tray]

        self.canvas.fig.clear()
        ax1, ax2, ax3 = self.canvas.fig.subplots(3, 1, sharex=True)

        for cell, grp in sub.groupby('cell_no'):
            if int(cell) not in checked:
                continue
            t            = grp['t_sec'].values / 60
            color, alpha = self._get_cell_color(tray, int(cell))
            lw           = 1.0 if color == 'red' else 0.7
            zorder       = 3   if color == 'red' else 2

            l1, = ax1.plot(t, grp['current_A'].values * 1e6,
                           color=color, alpha=alpha, linewidth=lw,
                           zorder=zorder, picker=5)
            l1._cell_no = int(cell)

            l2, = ax2.plot(t, grp['voltage_V'].values * 1000,
                           color=color, alpha=alpha, linewidth=lw,
                           zorder=zorder, picker=5)
            l2._cell_no = int(cell)

            l3, = ax3.plot(t, grp['temp_C'].values,
                           color=color, alpha=alpha, linewidth=lw,
                           zorder=zorder, picker=5)
            l3._cell_no = int(cell)

            for ln in (l1, l2, l3):
                ln._orig_color  = color
                ln._orig_alpha  = alpha
                ln._orig_lw     = lw
                ln._orig_zorder = zorder
            self._line_refs[int(cell)] = [l1, l2, l3]

        has_grade = (not self.state.df_meta.empty and
                     PROCESS_COL_GRADE in self.state.df_meta.columns)
        legend_note = '  (빨=E등급)' if has_grade else ''

        ax1.set_ylabel('전류 (µA)')
        ax1.set_title(f'{tray} — 전체 셀 오버레이{legend_note}')
        ax1.grid(True, alpha=0.3)
        ax2.set_ylabel('전압 (mV)')
        ax2.grid(True, alpha=0.3)
        ax3.set_ylabel('온도 (°C)')
        ax3.set_xlabel('시간 (분)')
        ax3.grid(True, alpha=0.3)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _on_pick(self, event):
        cell_no = getattr(event.artist, '_cell_no', None)
        if cell_no is None:
            return

        # 이전 하이라이트 복원
        if self._highlighted_cell is not None:
            for ln in self._line_refs.get(self._highlighted_cell, []):
                ln.set_color(ln._orig_color)
                ln.set_alpha(ln._orig_alpha)
                ln.set_linewidth(ln._orig_lw)
                ln.set_zorder(ln._orig_zorder)

        # 새 하이라이트 적용
        for ln in self._line_refs.get(cell_no, []):
            ln.set_color('royalblue')
            ln.set_alpha(1.0)
            ln.set_linewidth(2.5)
            ln.set_zorder(10)

        self._highlighted_cell = cell_no
        label = cell_to_label(cell_no)
        self.lbl_picked.setText(f'선택: {cell_no}번  ({label})')

        axes = self.canvas.fig.get_axes()
        if axes:
            tray = self.cb_tray.currentText()
            has_grade = (not self.state.df_meta.empty and
                         PROCESS_COL_GRADE in self.state.df_meta.columns)
            legend_note = '  (빨=E등급)' if has_grade else ''
            axes[0].set_title(
                f'{tray} — 전체 셀 오버레이{legend_note}'
                f'  │  선택: {cell_no}번 ({label})'
            )

        self.canvas.draw_idle()

    def _plot_cell(self):
        tray      = self.cb_tray.currentText()
        cell_text = self.cb_cell.currentText()
        if not tray or not cell_text:
            return
        cell = int(cell_text)
        sub  = self.state.df_ts[
            (self.state.df_ts['tray_id'] == tray) &
            (self.state.df_ts['cell_no']  == cell)
        ]
        if sub.empty:
            return

        self.canvas.fig.clear()
        ax1, ax2, ax3 = self.canvas.fig.subplots(3, 1, sharex=True)

        t = sub['t_sec'].values / 60
        ax1.plot(t, sub['current_A'].values * 1e6, color='steelblue')
        ax1.set_ylabel('전류 (µA)')
        ax1.set_title(f'Tray: {tray}  |  Cell: {cell}  ({cell_to_label(cell)})')
        ax1.grid(True, alpha=0.3)

        ax2.plot(t, sub['voltage_V'].values * 1000, color='orange')
        ax2.set_ylabel('전압 (mV)')
        ax2.grid(True, alpha=0.3)

        ax3.plot(t, sub['temp_C'].values, color='red')
        ax3.set_ylabel('온도 (°C)')
        ax3.set_xlabel('시간 (분)')
        ax3.grid(True, alpha=0.3)

        self.canvas.fig.tight_layout()
        self.canvas.draw()
