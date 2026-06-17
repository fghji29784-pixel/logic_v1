# ─────────────────────────────────────────────
#  main.py  —  SDM 분석 툴 (PySide6 + Matplotlib)
#  실행: python main.py
#  배포: pyinstaller --onefile --windowed main.py
# ─────────────────────────────────────────────

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
import matplotlib.colors as mcolors
import seaborn as sns

from parser   import parse_multiple_trays, get_layer
from analysis import (
    add_layer_dummies, run_analysis, run_analysis_per_tray,
    separation_curve, confusion_at_threshold, compute_metrics,
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


# ══════════════════════════════════════════════
#  공유 상태 (AppState)
# ══════════════════════════════════════════════

class AppState:
    def __init__(self):
        self.df_meta: pd.DataFrame     = pd.DataFrame()
        self.df_ts:   pd.DataFrame     = pd.DataFrame()
        self.n_minutes: int            = 15
        self.analysis_results: dict    = {}   # option → result dict
        self.selected_option: int      = 1
        self.dep_type: str             = 'single'
        self.threshold: float          = 2.0
        self.rwiring_threshold: float | None = None
        self.feature_list: list | None = None   # 보정 변수 선택 (None = 기본값)
        self.callbacks: list           = []   # 데이터 갱신 시 호출

    def notify(self):
        for cb in self.callbacks:
            try:
                cb()
            except Exception as e:
                print(f'[콜백 오류] {e}')


# ══════════════════════════════════════════════
#  분석 백그라운드 스레드
# ══════════════════════════════════════════════

class AnalysisWorker(QThread):
    finished = Signal(dict)
    error    = Signal(str)

    def __init__(self, state: AppState, option: int,
                 feature_list: list | None = None):
        super().__init__()
        self.state        = state
        self.option       = option
        self.feature_list = feature_list

    def run(self):
        try:
            res = run_analysis_per_tray(
                self.state.df_meta,
                option=self.option,
                n_minutes=self.state.n_minutes,
                dep_type=self.state.dep_type,
                rwiring_threshold=self.state.rwiring_threshold,
                feature_list=self.feature_list,
            )
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


# ══════════════════════════════════════════════
#  공통 Matplotlib 캔버스 위젯
# ══════════════════════════════════════════════

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(6, 4)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

    def clear(self):
        self.fig.clear()
        self.draw()


# ══════════════════════════════════════════════
#  Tab 1: 데이터 불러오기
# ══════════════════════════════════════════════

class Tab1Load(QWidget):
    data_loaded = Signal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── 상위 폴더 선택 ──
        g_folder = QGroupBox('① SDM 데이터 폴더 선택')
        fl = QVBoxLayout(g_folder)

        h1 = QHBoxLayout()
        self.btn_folder = QPushButton('상위 폴더 선택…')
        self.lbl_folder = QLabel('(선택 없음)')
        self.lbl_folder.setWordWrap(True)
        h1.addWidget(self.btn_folder)
        h1.addWidget(self.lbl_folder, 1)
        fl.addLayout(h1)

        fl.addWidget(QLabel('트레이 목록 (복수 선택 가능):'))
        self.lst_trays = QListWidget()
        self.lst_trays.setSelectionMode(QAbstractItemView.MultiSelection)
        fl.addWidget(self.lst_trays)
        layout.addWidget(g_folder)

        # ── 공정 데이터 ──
        g_proc = QGroupBox('② 공정 데이터 파일 선택 (dOCV 파일)')
        pl = QHBoxLayout(g_proc)
        self.btn_proc = QPushButton('파일 선택…')
        self.lbl_proc = QLabel('(선택 없음)')
        self.lbl_proc.setWordWrap(True)
        pl.addWidget(self.btn_proc)
        pl.addWidget(self.lbl_proc, 1)
        layout.addWidget(g_proc)

        # ── 측정 시간 N ──
        g_n = QGroupBox('③ 측정 시간 N (분)')
        nl = QHBoxLayout(g_n)
        self.spin_n = QSpinBox()
        self.spin_n.setRange(5, 30)
        self.spin_n.setValue(15)
        nl.addWidget(self.spin_n)
        nl.addStretch()
        layout.addWidget(g_n)

        # ── 불러오기 버튼 ──
        self.btn_load = QPushButton('▶  데이터 불러오기')
        self.btn_load.setFixedHeight(40)
        font = self.btn_load.font()
        font.setBold(True)
        self.btn_load.setFont(font)
        layout.addWidget(self.btn_load)

        # ── 미리보기 테이블 ──
        layout.addWidget(QLabel('미리보기 (df_meta 상위 50행):'))
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.preview_table)

        # 시그널
        self.btn_folder.clicked.connect(self._select_folder)
        self.btn_proc.clicked.connect(self._select_proc)
        self.btn_load.clicked.connect(self._load_data)

        self._folder_path   = None
        self._proc_filepath = None

    def _select_folder(self):
        path = QFileDialog.getExistingDirectory(self, '상위 폴더 선택')
        if path:
            self._folder_path = path
            self.lbl_folder.setText(path)
            self._scan_trays(path)

    def _scan_trays(self, root: str):
        self.lst_trays.clear()
        p = Path(root)
        subdirs = [d for d in sorted(p.iterdir()) if d.is_dir()]
        for d in subdirs:
            has_kss  = any(d.glob('*.kss')) or any(d.glob('*.KSS'))
            has_temp = any(d.glob('*TEMP_DATA*')) or any(d.glob('*TEMP*'))
            if has_kss:
                item = QListWidgetItem(f'{d.name}{"  [온도O]" if has_temp else "  [온도X]"}')
                item.setData(Qt.UserRole, str(d))
                self.lst_trays.addItem(item)

    def _select_proc(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '공정 데이터 파일 선택', '',
            'Excel/CSV (*.xlsx *.xls *.csv);;모든 파일 (*)'
        )
        if path:
            self._proc_filepath = path
            self.lbl_proc.setText(Path(path).name)

    def _load_data(self):
        selected = self.lst_trays.selectedItems()
        if not selected:
            QMessageBox.warning(self, '경고', '트레이를 1개 이상 선택하세요.')
            return
        folder_paths = [item.data(Qt.UserRole) for item in selected]
        n = self.spin_n.value()

        dlg = QProgressDialog('데이터 불러오는 중…', None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()
        QApplication.processEvents()

        try:
            df_meta, df_ts = parse_multiple_trays(
                folder_paths,
                process_filepath=self._proc_filepath,
                n_minutes=n,
            )
            self.state.df_meta   = df_meta
            self.state.df_ts     = df_ts
            self.state.n_minutes = n
            self.state.analysis_results = {}
            self._show_preview(df_meta)
            self.state.notify()
            self.data_loaded.emit()
            dlg.close()
            QMessageBox.information(self, '완료',
                f'셀 {len(df_meta)}개 불러오기 완료.\n'
                f'시계열 행 수: {len(df_ts):,}')
        except Exception as e:
            dlg.close()
            QMessageBox.critical(self, '오류', str(e))

    def _show_preview(self, df: pd.DataFrame):
        sub = df.head(50)
        self.preview_table.setRowCount(len(sub))
        self.preview_table.setColumnCount(len(sub.columns))
        self.preview_table.setHorizontalHeaderLabels(list(sub.columns))
        for r, row in enumerate(sub.itertuples(index=False)):
            for c, val in enumerate(row):
                self.preview_table.setItem(r, c, QTableWidgetItem(str(val)))
        self.preview_table.resizeColumnsToContents()


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
            f'SDM 전류 (i_{15}min)', '온도 (T_final)',
            'Rwiring', 'dOCV (#07)', '판정등급',
        ])
        ctrl.addWidget(self.cb_type)

        self.chk_clip = QCheckBox('클리핑')
        self.chk_clip.setChecked(True)
        self.chk_clip.setToolTip('상위 N% 초과값을 colormap 최대값으로 클리핑')
        ctrl.addWidget(self.chk_clip)
        self.spin_clip = QSpinBox()
        self.spin_clip.setRange(80, 100)
        self.spin_clip.setValue(99)
        self.spin_clip.setSuffix('%')
        self.spin_clip.setFixedWidth(60)
        ctrl.addWidget(self.spin_clip)

        self.btn_draw = QPushButton('그리기')
        ctrl.addWidget(self.btn_draw)
        self.btn_save_all = QPushButton('전체 저장 (PNG)')
        ctrl.addWidget(self.btn_save_all)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.canvas = PlotCanvas(figsize=(8, 7))
        layout.addWidget(self.canvas)

        self.btn_draw.clicked.connect(self._draw)
        self.btn_save_all.clicked.connect(self._save_all)

    def _refresh(self):
        self.cb_tray.clear()
        if self.state.df_meta.empty:
            return
        trays = self.state.df_meta['tray_id'].unique().tolist()
        self.cb_tray.addItems(trays)
        # cb_type의 SDM 전류 항목 이름 업데이트
        n = self.state.n_minutes
        self.cb_type.setItemText(0, f'SDM 전류 (i_{n}min)')

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

    def _draw(self):
        tray = self.cb_tray.currentText()
        kind = self.cb_type.currentIndex()
        if not tray or self.state.df_meta.empty:
            return
        self._draw_core(self.canvas.fig, tray, kind)
        self.canvas.draw()

    def _draw_core(self, fig, tray: str, kind: int):
        n   = self.state.n_minutes
        col_map = {
            0: (f'i_{n}min', f'SDM 전류 (µA)',  lambda v: v * 1e6),
            1: ('t_final',   '온도 (°C)',         lambda v: v),
            2: ('rwiring',   'Rwiring (Ω)',        lambda v: v),
            3: (PROCESS_COL_DOCV, 'dOCV #07',     lambda v: v),
            4: (PROCESS_COL_GRADE, '판정등급',     None),
        }
        col, title, transform = col_map[kind]

        fig.clear()
        ax = fig.add_subplot(111)

        row_labels = [chr(ord('A') + i) for i in range(TRAY_COLS)]
        col_labels  = list(range(1, TRAY_ROWS + 1))

        if kind == 4:
            sub = self.state.df_meta[self.state.df_meta['tray_id'] == tray]
            grid = np.full((TRAY_ROWS, TRAY_COLS), np.nan)
            for _, row in sub.iterrows():
                cn    = int(row['cell_no']) - 1
                x_idx = cn // TRAY_COLS
                y_idx = cn %  TRAY_COLS
                grade = str(row.get(PROCESS_COL_GRADE, '')).upper()
                grid[y_idx, x_idx] = 1.0 if grade == 'E' else 0.0
            cmap = mcolors.ListedColormap(['black', 'red'])
            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect='equal')
            ax.set_title(f'{tray} — 판정등급 (검=A, 빨=E)')
            for iy in range(TRAY_ROWS):
                for ix in range(TRAY_COLS):
                    val = grid[iy, ix]
                    if not np.isnan(val):
                        ax.text(ix, iy, 'E' if val == 1.0 else 'A',
                                ha='center', va='center', fontsize=6,
                                color='white', fontweight='bold')
        else:
            grid = self._get_grid(tray, col)
            if transform:
                grid = transform(grid)
            vmax_val = None
            if self.chk_clip.isChecked():
                flat = grid[~np.isnan(grid)]
                if len(flat) > 0:
                    vmax_val = np.percentile(flat, self.spin_clip.value())
            im = ax.imshow(grid, cmap='RdYlGn_r', aspect='equal', vmax=vmax_val)
            fig.colorbar(im, ax=ax)
            clip_note = (f'  (상위 {self.spin_clip.value()}% 클리핑)'
                         if self.chk_clip.isChecked() else '')
            ax.set_title(f'{tray} — {title}{clip_note}')
            fmt_map = {0: '{:.2f}', 1: '{:.1f}', 2: '{:.4f}', 3: '{:.1f}'}
            fmt = fmt_map.get(kind, '{:.2f}')
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
                                ha='center', va='center', fontsize=5.5, color=tc)

        ax.set_xticks(range(TRAY_COLS))
        ax.set_yticks(range(TRAY_ROWS))
        ax.set_xticklabels(row_labels, fontsize=8)
        ax.set_yticklabels(col_labels, fontsize=8)
        ax.set_xlabel('행 (A–L)', fontsize=9)
        ax.set_ylabel('열 (1–12)', fontsize=9)
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
        kind_names = ['SDM전류', '온도', 'Rwiring', 'dOCV', '판정등급']
        count = 0
        for tray in trays:
            for kind, name in enumerate(kind_names):
                fig = plt.figure(figsize=(8, 7))
                self._draw_core(fig, tray, kind)
                fig.savefig(f'{folder}/{tray}_{name}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                count += 1
        QMessageBox.information(self, '완료', f'{count}개 히트맵 저장 완료\n{folder}')


# ══════════════════════════════════════════════
#  Tab 4: 분석 로직
# ══════════════════════════════════════════════

class Tab4Analysis(QWidget):
    analysis_done = Signal(int)  # option

    def __init__(self, state: AppState):
        super().__init__()
        self.state    = state
        self.workers: dict[int, AnalysisWorker] = {}
        self._cur_res: dict | None = None
        self._build_ui()
        state.callbacks.append(self._refresh)

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # ── 왼쪽: 설정 패널 ──
        ctrl = QWidget()
        ctrl.setFixedWidth(260)
        cl = QVBoxLayout(ctrl)

        # 옵션 선택
        g_opt = QGroupBox('모델 옵션')
        ol = QVBoxLayout(g_opt)
        self.opt_group = QButtonGroup(self)
        opt_info = [
            ('옵션 1: OLS (SDM만)',         1,
             'OLS 선형회귀. SDM 변수(초기전압·온도·위치)만으로 전류값 보정.\n가장 단순한 기준선 모델.'),
            ('옵션 2: OLS (SDM+공정)',       2,
             'OLS 선형회귀 + 공정 OCV·충전전압 추가.\n공정 데이터가 보정에 도움이 되는지 확인할 때 사용.'),
            ('옵션 3: Robust (SDM만)',       3,
             'Robust 회귀. 잔차 상·하위 10% 셀을 제거 후 재추정(WLS).\n불량셀이 회귀계수를 왜곡하지 않도록 방어. SDM 변수만 사용.'),
            ('옵션 4: Robust (SDM+공정)',    4,
             'Robust 회귀 + 공정 변수.\n옵션 3의 이상치 내성에 공정 데이터까지 추가.'),
            ('옵션 5: Lasso + LOO-tray CV', 5,
             'Lasso 회귀 + 트레이 단위 교차검증(LOO-tray CV).\n불필요한 공정 변수를 자동으로 0으로 줄임.\n다중공선성·과적합을 동시에 방어.'),
        ]
        for txt, val, tip in opt_info:
            rb = QRadioButton(txt)
            rb.setToolTip(tip)
            if val == 1:
                rb.setChecked(True)
            self.opt_group.addButton(rb, val)
            ol.addWidget(rb)
        _lbl = QLabel("※ 마우스를 올리면 옵션 설명 표시\n   옵션 1~5 전부 실행 후 d' 비교 권장")
        _lbl.setWordWrap(True)
        _lbl.setStyleSheet('color:#777; font-size:10px;')
        ol.addWidget(_lbl)
        cl.addWidget(g_opt)

        # 종속변수
        g_dep = QGroupBox('종속변수')
        dl = QVBoxLayout(g_dep)
        self.rb_single = QRadioButton('단일값 (v1)')
        self.rb_single.setToolTip('N분 시점의 전류값 한 개를 종속변수로 사용.\n구현이 단순하고 직관적.')
        self.rb_slope  = QRadioButton('기울기 slope (v2)')
        self.rb_slope.setToolTip('0~N분 구간 전류 기울기(ΔI/Δt)를 종속변수로 사용.\n순간 노이즈에 덜 민감하여 더 안정적.')
        self.rb_single.setChecked(True)
        dl.addWidget(self.rb_single)
        dl.addWidget(self.rb_slope)
        cl.addWidget(g_dep)

        # Rwiring 임계값
        g_rw = QGroupBox('Rwiring 임계값 (0 = 미적용)')
        g_rw.setToolTip('배선·접촉 저항(Ω). 이 값 초과 채널은 접촉 불량으로 판단해 분석에서 제외.\n0이면 모든 채널 포함.')
        rl = QHBoxLayout(g_rw)
        self.spin_rw = QDoubleSpinBox()
        self.spin_rw.setRange(0, 100)
        self.spin_rw.setValue(0)
        self.spin_rw.setSuffix(' Ω')
        self.spin_rw.setToolTip('히트맵(Rwiring)에서 이상치로 보이는 채널 값을 기준으로 설정.')
        rl.addWidget(self.spin_rw)
        cl.addWidget(g_rw)

        # 보정 변수 선택
        g_vars = QGroupBox('보정 변수 선택')
        vl = QVBoxLayout(g_vars)

        vl.addWidget(QLabel('SDM 계열 (기본 체크):'))
        sdm_grid = QGridLayout()
        self.var_checks = {}
        sdm_items = [
            ('v_init',    'v_init (초기전압)',
             '측정 시작 전압(mV). SOC 차이로 인한 전류 편차를 보정.'),
            ('t_final',   't_final (최종온도)',
             '측정 종료 온도(℃). 온도가 높을수록 자가방전 전류가 커지므로 보정 필수.'),
            ('delta_t',   'ΔT (온도변화)',
             '측정 중 온도 변화량(℃). 측정 도중 온도 드리프트를 보정.'),
            ('layer_pos', '레이어위치 (L2~L6)',
             '트레이 내 위치 더미변수. 가장자리(L1) 기준, 중앙(L6)으로 갈수록 온도 높음.\n위치별 온도·저항 불균일 보정.'),
        ]
        for idx, (key, label, tip) in enumerate(sdm_items):
            cb = QCheckBox(label)
            cb.setToolTip(tip)
            cb.setChecked(True)
            self.var_checks[key] = cb
            sdm_grid.addWidget(cb, idx // 2, idx % 2)
        vl.addLayout(sdm_grid)

        vl.addWidget(QLabel('공정 계열 (옵션 2/4/5에서 사용):'))
        proc_grid = QGridLayout()
        proc_items = [
            ('OCV1',         'OCV1',         '에이징 1단계 OCV. 초기 셀 상태 반영.'),
            ('OCV2',         'OCV2',         '에이징 2단계 OCV.'),
            ('OCV3',         'OCV3',         '에이징 3단계 OCV.'),
            ('OCV4',         'OCV4',         '에이징 4단계 OCV.'),
            ('OCV7',         'OCV7',         '에이징 7단계 OCV. dOCV 계산 기준점.'),
            ('CHARGE_END_V', '1차충전종료전압', '1차 충전 종료 전압. 초기 셀 용량 차이 보정.'),
        ]
        for idx, (key, label, tip) in enumerate(proc_items):
            cb = QCheckBox(label)
            cb.setToolTip(tip)
            cb.setChecked(False)
            self.var_checks[key] = cb
            proc_grid.addWidget(cb, idx // 2, idx % 2)
        vl.addLayout(proc_grid)
        _lbl2 = QLabel('※ VIF > 10 경고 시 해당 변수 해제\n   또는 옵션 5(Lasso) 사용 권장')
        _lbl2.setWordWrap(True)
        _lbl2.setStyleSheet('color:#777; font-size:10px;')
        vl.addWidget(_lbl2)
        cl.addWidget(g_vars)

        # 실행 버튼
        self.btn_run_one = QPushButton('▶  현재 옵션 실행')
        self.btn_run_all = QPushButton('▶▶  옵션 1~5 전부 실행')
        self.btn_run_one.setFixedHeight(36)
        self.btn_run_all.setFixedHeight(36)
        cl.addWidget(self.btn_run_one)
        cl.addWidget(self.btn_run_all)

        cl.addStretch()
        layout.addWidget(ctrl)

        # ── 오른쪽: 결과 출력 ──
        right = QWidget()
        rl2 = QVBoxLayout(right)

        # 회귀계수 테이블
        rl2.addWidget(QLabel('회귀계수 및 VIF:'))
        _d1 = QLabel('계수: 해당 변수가 전류값에 미치는 영향크기  |  '
                     'p-value < 0.05: 통계적으로 유의미  |  '
                     'VIF > 10(빨강): 다중공선성 → 해당 변수 해제 또는 옵션5 사용')
        _d1.setWordWrap(True)
        _d1.setStyleSheet('color:#777; font-size:10px;')
        rl2.addWidget(_d1)
        self.tbl_coef = QTableWidget(0, 4)
        self.tbl_coef.setHorizontalHeaderLabels(['변수', '계수', 'p-value', 'VIF'])
        self.tbl_coef.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_coef.setFixedHeight(200)
        rl2.addWidget(self.tbl_coef)

        # R² 라벨
        self.lbl_r2 = QLabel('R² : —   Adj.R² : —')
        self.lbl_r2.setToolTip(
            'R²: 독립변수들이 전류값 분산을 설명하는 비율 (0~1). 높을수록 보정 잘 됨.\n'
            'Adj.R²: 변수 수 패널티 포함. 변수 추가 후에도 올라가야 의미 있음.')
        rl2.addWidget(self.lbl_r2)

        # 옵션별 분리도 비교
        rl2.addWidget(QLabel('옵션별 분리도 비교:'))
        _d2 = QLabel("d'(분리도): 불량·양품 분포 간 거리(σ 단위). 2 이상이면 실용 가능, 높을수록 좋음.  |  "
                     "AUC: 기준선 위치와 무관한 전체 판별력. 1에 가까울수록 좋음(0.9 이상 목표).")
        _d2.setWordWrap(True)
        _d2.setStyleSheet('color:#777; font-size:10px;')
        rl2.addWidget(_d2)
        self.tbl_sep = QTableWidget(5, 3)
        self.tbl_sep.setHorizontalHeaderLabels(['옵션', "d'", 'AUC'])
        self.tbl_sep.setFixedHeight(140)
        for i in range(5):
            self.tbl_sep.setItem(i, 0, QTableWidgetItem(f'옵션 {i+1}'))
        rl2.addWidget(self.tbl_sep)

        # 보정값 분포 그래프
        _d3 = QLabel('보정값 분포: 외부 요인(온도·전압·위치) 제거 후 남은 순수 자가방전 편차 분포.\n'
                     '양품 셀이 좁게 모이고 불량 셀이 오른쪽 꼬리에 분리될수록 판정에 유리.')
        _d3.setWordWrap(True)
        _d3.setStyleSheet('color:#777; font-size:10px;')
        rl2.addWidget(_d3)

        tray_row = QHBoxLayout()
        tray_row.addWidget(QLabel('트레이 선택:'))
        self.cb_tray = QComboBox()
        self.cb_tray.addItem('전체 (합산)')
        self.cb_tray.setToolTip('전체: 전 트레이 합산 분포\n개별 트레이: 해당 트레이만의 분포 (트레이별 z-score 기준)')
        tray_row.addWidget(self.cb_tray)
        tray_row.addStretch()
        rl2.addLayout(tray_row)

        self.canvas = PlotCanvas(figsize=(8, 3))
        rl2.addWidget(self.canvas)

        layout.addWidget(right)

        # 시그널
        self.btn_run_one.clicked.connect(self._run_one)
        self.btn_run_all.clicked.connect(self._run_all)
        self.cb_tray.currentIndexChanged.connect(self._update_tray_display)

    def _refresh(self):
        self.tbl_coef.setRowCount(0)
        self.lbl_r2.setText('R² : —   Adj.R² : —')

    def _get_settings(self):
        opt      = self.opt_group.checkedId()
        dep_type = 'slope' if self.rb_slope.isChecked() else 'single'
        rw       = self.spin_rw.value() or None
        checked  = [k for k, cb in self.var_checks.items() if cb.isChecked()]
        feature_list = checked if checked else None   # 전부 해제 시 기본 동작 폴백
        return opt, dep_type, rw, feature_list

    def _run_one(self):
        if self.state.df_meta.empty:
            QMessageBox.warning(self, '경고', '먼저 데이터를 불러오세요.')
            return
        opt, dep_type, rw, feature_list = self._get_settings()
        self.state.dep_type           = dep_type
        self.state.rwiring_threshold  = rw
        self.state.feature_list       = feature_list
        self._launch_worker(opt, feature_list)

    def _run_all(self):
        if self.state.df_meta.empty:
            QMessageBox.warning(self, '경고', '먼저 데이터를 불러오세요.')
            return
        _, dep_type, rw, feature_list = self._get_settings()
        self.state.dep_type          = dep_type
        self.state.rwiring_threshold = rw
        self.state.feature_list      = feature_list
        for opt in range(1, 6):
            self._launch_worker(opt, feature_list)

    def _launch_worker(self, opt: int, feature_list: list | None = None):
        w = AnalysisWorker(self.state, opt, feature_list=feature_list)
        self.workers[opt] = w
        w.finished.connect(lambda res, o=opt: self._on_done(o, res))
        w.error.connect(lambda e, o=opt:
                        QMessageBox.critical(self, f'옵션 {o} 오류', e))
        w.start()

    def _on_done(self, opt: int, res: dict):
        self.state.analysis_results[opt] = res
        self.analysis_done.emit(opt)

        # 현재 선택 옵션이면 화면 업데이트
        if opt == self.opt_group.checkedId():
            self._display(res)

        # 분리도 비교표 업데이트
        m = res.get('metrics', {})
        self.tbl_sep.setItem(opt - 1, 1,
            QTableWidgetItem(f"{m.get('d_prime', '—'):.3f}"
                             if isinstance(m.get('d_prime'), float) else '—'))
        self.tbl_sep.setItem(opt - 1, 2,
            QTableWidgetItem(f"{m.get('auc', '—'):.3f}"
                             if isinstance(m.get('auc'), float) else '—'))

    def _display(self, res: dict):
        self._cur_res = res
        # 트레이 목록 갱신 (선택 유지)
        prev = self.cb_tray.currentText()
        self.cb_tray.blockSignals(True)
        self.cb_tray.clear()
        self.cb_tray.addItem('전체 (합산)')
        for tid in res.get('per_tray_results', {}):
            self.cb_tray.addItem(str(tid))
        idx = self.cb_tray.findText(prev)
        self.cb_tray.setCurrentIndex(idx if idx >= 0 else 0)
        self.cb_tray.blockSignals(False)
        self._update_tray_display()

    def _update_tray_display(self):
        """트레이 선택에 따라 회귀계수 테이블 + 히스토그램 동시 갱신."""
        res = self._cur_res
        if res is None:
            return

        sel          = self.cb_tray.currentText()
        per_tray_res = res.get('per_tray_results', {})

        if sel == '전체 (합산)' or sel not in per_tray_res:
            model        = res.get('model')
            vif          = res.get('vif')   # 트레이별 평균 VIF
            corrected    = res.get('corrected')
            df_valid     = res.get('df_valid', pd.DataFrame())
            title_suffix = '전체'
            r2_note = ''
            if res.get('per_tray'):
                r2_note = f'  [트레이별 ×{res.get("n_trays","?")}  대표=첫 트레이]'
                if res.get('lasso_fallback'):
                    r2_note += '  ※옵션5→OLS'
        else:
            t_res        = per_tray_res[sel]
            model        = t_res.get('model')
            vif          = t_res.get('vif')
            corrected    = t_res.get('corrected')
            df_valid     = t_res.get('df_valid', pd.DataFrame())
            title_suffix = sel
            r2_note      = f'  [트레이: {sel}]'

        # ── 회귀계수 테이블 ──
        self.tbl_coef.setRowCount(0)
        if model is not None and hasattr(model, 'params'):
            params  = model.params.drop('const', errors='ignore')
            pvals   = model.pvalues.drop('const', errors='ignore')
            vif_map = {}
            if vif is not None and isinstance(vif, pd.DataFrame):
                vif_map = dict(zip(vif['feature'], vif['VIF']))
            for feat in params.index:
                r = self.tbl_coef.rowCount()
                self.tbl_coef.insertRow(r)
                self.tbl_coef.setItem(r, 0, QTableWidgetItem(feat))
                self.tbl_coef.setItem(r, 1, QTableWidgetItem(f'{params[feat]:.4f}'))
                self.tbl_coef.setItem(r, 2, QTableWidgetItem(
                    f'{pvals.get(feat, float("nan")):.4f}'))
                vif_val  = vif_map.get(feat, float('nan'))
                item_vif = QTableWidgetItem(f'{vif_val:.1f}')
                if vif_val > 10:
                    item_vif.setBackground(QColor(255, 200, 200))
                self.tbl_coef.setItem(r, 3, item_vif)
            r2  = getattr(model, 'rsquared', float('nan'))
            ar2 = getattr(model, 'rsquared_adj', float('nan'))
            self.lbl_r2.setText(f'R² : {r2:.4f}   Adj.R² : {ar2:.4f}{r2_note}')

        # ── 보정값 분포 히스토그램 ──
        if corrected is None:
            return
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        if PROCESS_COL_GRADE in df_valid.columns:
            grades = df_valid[PROCESS_COL_GRADE].astype(str).str.strip().str.upper()
            good_c = corrected[grades == 'A'].dropna()
            bad_c  = corrected[grades == 'E'].dropna()
            other  = corrected[~grades.isin(['A', 'E'])].dropna()
            if not other.empty:
                ax.hist(other, bins=40, color='#888', alpha=0.5, edgecolor='white', label='미분류')
            if not good_c.empty:
                ax.hist(good_c, bins=40, color='steelblue', alpha=0.6,
                        edgecolor='white', label='양품(A)')
            if not bad_c.empty:
                ax.hist(bad_c, bins=10, color='red', alpha=0.85,
                        edgecolor='white', label='불량(E)')
            ax.legend(fontsize=8)
        else:
            ax.hist(corrected.dropna(), bins=40, edgecolor='white', alpha=0.7)
        ax.set_xlabel('보정값 (µA)')
        ax.set_ylabel('빈도')
        ax.set_title(f'옵션 {res["option"]} 보정값 분포 [{title_suffix}]  (파랑=양품A  빨강=불량E)')
        ax.grid(True, alpha=0.3)
        self.canvas.fig.tight_layout()
        self.canvas.draw()


# ══════════════════════════════════════════════
#  Tab 5: 결과 및 판정
# ══════════════════════════════════════════════

class Tab5Result(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._build_ui()
        state.callbacks.append(self._refresh)

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # ── 왼쪽 설정 ──
        ctrl = QWidget()
        ctrl.setFixedWidth(220)
        cl = QVBoxLayout(ctrl)

        cl.addWidget(QLabel('옵션 선택:'))
        self.cb_opt = QComboBox()
        self.cb_opt.addItems([
            '옵션 1: OLS (SDM만)',
            '옵션 2: OLS (SDM+공정)',
            '옵션 3: Robust (SDM만)',
            '옵션 4: Robust (SDM+공정)',
            '옵션 5: Lasso+CV',
        ])
        cl.addWidget(self.cb_opt)

        cl.addWidget(QLabel('트레이 선택:'))
        self.cb_tray = QComboBox()
        self.cb_tray.addItem('전체 (합산)')
        self.cb_tray.setToolTip(
            '전체: 전 트레이 합산 기준 z-score/혼동행렬/분리도 커브\n'
            '개별 트레이: 해당 트레이만의 트레이 내 상대 z-score 기준')
        cl.addWidget(self.cb_tray)

        cl.addWidget(QLabel('기준선 (z-score):'))
        _d_z = QLabel('보정값을 표준화한 점수.\n이 값 이상인 셀을 불량으로 판정.\n올릴수록 FP↓ FN↑, 내릴수록 FP↑ FN↓')
        _d_z.setWordWrap(True)
        _d_z.setStyleSheet('color:#777; font-size:10px;')
        cl.addWidget(_d_z)
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(-10, 20)
        self.spin_thresh.setValue(2.0)
        self.spin_thresh.setSingleStep(0.1)
        self.spin_thresh.setToolTip(
            'z-score 기준선. 이 값 이상인 셀 → 불량 판정.\n'
            '오른쪽으로 높일수록: FP 감소(과검출 줄어듦), FN 증가(불량 놓칠 위험).\n'
            '왼쪽으로 낮출수록: FP 증가, FN 감소.\n'
            '일반적으로 FN=0(불량 전량 검출) 유지하면서 FP를 최소화하는 값을 선택.')
        cl.addWidget(self.spin_thresh)

        self.btn_draw = QPushButton('그래프 갱신')
        cl.addWidget(self.btn_draw)

        # TP/FP/FN/TN
        g_cm = QGroupBox('혼동행렬 (판정등급 있을 때만 표시)')
        gml  = QGridLayout(g_cm)
        self.lbl_tp = QLabel('TP(정검출): —')
        self.lbl_fp = QLabel('FP(과검출): —')
        self.lbl_fn = QLabel('FN(미검출): —')
        self.lbl_tn = QLabel('TN(정통과): —')
        self.lbl_tp.setToolTip('True Positive: 실제 불량(E)을 불량으로 올바르게 검출')
        self.lbl_fp.setToolTip('False Positive: 실제 양품(A)을 불량으로 잘못 판정 (과검출)')
        self.lbl_fn.setToolTip('False Negative: 실제 불량(E)을 양품으로 놓침 → 0이 목표')
        self.lbl_tn.setToolTip('True Negative: 실제 양품(A)을 양품으로 올바르게 통과')
        gml.addWidget(self.lbl_tp, 0, 0)
        gml.addWidget(self.lbl_fp, 0, 1)
        gml.addWidget(self.lbl_fn, 1, 0)
        gml.addWidget(self.lbl_tn, 1, 1)
        _d_cm = QLabel('목표: FN=0 유지하면서 FP 최소화')
        _d_cm.setStyleSheet('color:#777; font-size:10px;')
        gml.addWidget(_d_cm, 2, 0, 1, 2)
        cl.addWidget(g_cm)

        # d_prime / AUC
        self.lbl_dp  = QLabel("d'(분리도, ≥2 목표): —")
        self.lbl_auc = QLabel('AUC(판별력, ≥0.9 목표): —')
        self.lbl_dp.setToolTip("분리도. 불량 셀이 양품 분포에서 몇 σ 떨어져 있는지.\n2 이상이면 실용 가능, 높을수록 판별이 쉬움.")
        self.lbl_auc.setToolTip('ROC 곡선 면적. 기준선 위치와 무관한 전체 판별력.\n0.9 이상 목표. 1.0이면 완벽한 분리.')
        cl.addWidget(self.lbl_dp)
        cl.addWidget(self.lbl_auc)
        _d_dp = QLabel("d' ≥ 2, AUC ≥ 0.9 를 목표로 옵션·변수 조합 조정")
        _d_dp.setWordWrap(True)
        _d_dp.setStyleSheet('color:#777; font-size:10px;')
        cl.addWidget(_d_dp)

        cl.addStretch()
        layout.addWidget(ctrl)

        # ── 오른쪽 캔버스 (2×2) ──
        self.canvas = PlotCanvas(figsize=(10, 8))
        layout.addWidget(self.canvas)

        self.cb_opt.currentIndexChanged.connect(self._draw)
        self.cb_tray.currentIndexChanged.connect(self._draw)
        # spin_thresh → state.threshold 동기화 + 그래프 갱신
        self.spin_thresh.valueChanged.connect(
            lambda v: (setattr(self.state, 'threshold', v), self._draw())
        )
        self.btn_draw.clicked.connect(self._draw)

    def _refresh(self):
        pass  # 분석 완료 후 Tab4에서 analysis_done 시그널로 연결

    def _draw(self):
        opt = self.cb_opt.currentIndex() + 1
        res = self.state.analysis_results.get(opt)
        if res is None:
            return

        # ── 트레이 목록 갱신 (선택 유지) ──
        prev_sel = self.cb_tray.currentText()
        self.cb_tray.blockSignals(True)
        self.cb_tray.clear()
        self.cb_tray.addItem('전체 (합산)')
        for tid in res.get('per_tray_results', {}):
            self.cb_tray.addItem(str(tid))
        idx = self.cb_tray.findText(prev_sel)
        self.cb_tray.setCurrentIndex(idx if idx >= 0 else 0)
        self.cb_tray.blockSignals(False)

        sel          = self.cb_tray.currentText()
        per_tray_res = res.get('per_tray_results', {})

        # ── 트레이별 vs 전체 데이터 선택 ──
        if sel != '전체 (합산)' and sel in per_tray_res:
            t_res       = per_tray_res[sel]
            z_scores    = t_res.get('z_scores')
            corrected   = t_res.get('corrected')
            df_valid    = t_res.get('df_valid', pd.DataFrame())
            df_curve    = self.state.df_meta[
                self.state.df_meta['tray_id'] == sel
            ] if 'tray_id' in self.state.df_meta.columns else self.state.df_meta
            sel_label   = sel
            disp_metrics = compute_metrics(
                z_scores,
                df_valid[PROCESS_COL_GRADE] if PROCESS_COL_GRADE in df_valid.columns else None
            )
        else:
            z_scores     = res.get('z_scores')
            corrected    = res.get('corrected')
            df_valid     = res.get('df_valid', pd.DataFrame())
            df_curve     = self.state.df_meta
            sel_label    = '전체'
            disp_metrics = res.get('metrics', {})

        threshold   = self.spin_thresh.value()
        grade_col   = PROCESS_COL_GRADE
        true_labels = df_valid[grade_col] if grade_col in df_valid.columns else None
        docv_col    = PROCESS_COL_DOCV
        docv_vals   = df_valid[docv_col].apply(pd.to_numeric, errors='coerce') \
                      if docv_col in df_valid.columns else None

        self.canvas.fig.clear()
        axes = self.canvas.fig.subplots(2, 2)

        # ── (0,0) z-score 분포 ──
        ax = axes[0, 0]
        if z_scores is not None:
            grade_str = true_labels.astype(str).str.upper() if true_labels is not None else None
            good_z = z_scores[grade_str == 'A'] if grade_str is not None else z_scores
            bad_z  = z_scores[grade_str == 'E'] if grade_str is not None else pd.Series(dtype=float)
            ax.hist(good_z.dropna(), bins=30, color='black', alpha=0.6, label='양품(A)')
            if not bad_z.empty:
                ax.hist(bad_z.dropna(), bins=30, color='red', alpha=0.7, label='불량(E)')
            ax.axvline(threshold, color='blue', linestyle='--', label=f'기준선 {threshold}')
            ax.set_xlabel('z-score')
            ax.set_ylabel('빈도')
            ax.set_title(f'z-score 분포 [{sel_label}]\n(검=양품A, 빨=불량E, 파선=기준선)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # ── (0,1) dOCV vs 보정값 산점도 ──
        ax = axes[0, 1]
        if corrected is not None and docv_vals is not None:
            if true_labels is not None:
                grade_s = true_labels.astype(str).str.strip().str.upper()
                mask_ga = (grade_s == 'A')
                mask_ge = (grade_s == 'E')
                mask_gu = ~(mask_ga | mask_ge)
                if mask_gu.any():
                    ax.scatter(docv_vals[mask_gu], corrected[mask_gu],
                               alpha=0.3, s=8, color='#888', label='미분류')
                if mask_ga.any():
                    ax.scatter(docv_vals[mask_ga], corrected[mask_ga],
                               alpha=0.4, s=8, color='steelblue', label='양품(A)')
                if mask_ge.any():
                    ax.scatter(docv_vals[mask_ge], corrected[mask_ge],
                               alpha=0.9, s=14, color='red', label='불량(E)', zorder=3)
                ax.legend(fontsize=8)
            else:
                ax.scatter(docv_vals, corrected, alpha=0.4, s=10, color='steelblue')
            ax.set_xlabel('dOCV #07')
            ax.set_ylabel('SDM 보정값 (µA)')
            ax.set_title(f'dOCV vs SDM 보정값 [{sel_label}]\n(우상향 직선에 가까울수록 SDM이 기존 방법 대체)')
            ax.grid(True, alpha=0.3)

        # ── (1,0) 분리도 N분 커브 (선택 트레이 기준) ──
        ax = axes[1, 0]
        try:
            curve_df = separation_curve(
                df_curve, option=opt,
                dep_type=self.state.dep_type,
                n_range=range(5, 16),
                rwiring_threshold=self.state.rwiring_threshold,
                feature_list=self.state.feature_list,
            )
            ax.plot(curve_df['n_minutes'], curve_df['d_prime'], marker='o')
            ax.axhline(2.0, color='red', linestyle='--', alpha=0.5, label="d'=2")
            ax.set_xlabel('N (분)')
            ax.set_ylabel("d'")
            ax.set_title(f"분리도 vs 측정 시간 [{sel_label}]\n(d'=2 최초 돌파 = 실용 최단 판정 시간)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        except Exception:
            ax.set_title('분리도 커브 계산 실패')

        # ── (1,1) 옵션별 분리도 막대 (전체 기준 고정) ──
        ax = axes[1, 1]
        opts = []
        dps  = []
        for o in range(1, 6):
            r = self.state.analysis_results.get(o)
            if r:
                dp = r.get('metrics', {}).get('d_prime', np.nan)
                opts.append(f'옵{o}')
                dps.append(dp if isinstance(dp, float) else np.nan)
        if opts:
            colors = ['steelblue'] * len(opts)
            colors[opt - 1] = 'orange'
            ax.bar(opts, dps, color=colors)
            ax.set_ylabel("d'")
            ax.set_title("옵션별 분리도 비교 [전체]\n(주황=현재 선택, 높을수록 판별력 우수)")
            ax.grid(True, alpha=0.3, axis='y')

        self.canvas.fig.tight_layout()
        self.canvas.draw()

        # ── 혼동행렬 / d' / AUC (선택 트레이 기준) ──
        if z_scores is not None and true_labels is not None:
            cm = confusion_at_threshold(z_scores, true_labels, threshold)
            self.lbl_tp.setText(f'TP: {cm["TP"]}')
            self.lbl_fp.setText(f'FP: {cm["FP"]}')
            self.lbl_fn.setText(f'FN: {cm["FN"]}')
            self.lbl_tn.setText(f'TN: {cm["TN"]}')

        self.lbl_dp.setText(f"d' : {disp_metrics.get('d_prime', '—'):.3f}"
                            if isinstance(disp_metrics.get('d_prime'), float) else "d' : —")
        self.lbl_auc.setText(f"AUC: {disp_metrics.get('auc', '—'):.3f}"
                             if isinstance(disp_metrics.get('auc'), float) else 'AUC: —')


# ══════════════════════════════════════════════
#  Tab 6: 통합 결과 테이블
# ══════════════════════════════════════════════

class Tab6Table(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._build_ui()
        state.callbacks.append(self._refresh)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel('옵션:'))
        self.cb_opt = QComboBox()
        self.cb_opt.addItems([f'옵션 {i}' for i in range(1, 6)])
        ctrl.addWidget(self.cb_opt)

        ctrl.addWidget(QLabel('등급 필터:'))
        self.cb_grade = QComboBox()
        self.cb_grade.addItems(['전체', 'A만', 'E만'])
        ctrl.addWidget(self.cb_grade)

        self.btn_build = QPushButton('테이블 생성')
        ctrl.addWidget(self.btn_build)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        layout.addWidget(self.table)

        self.btn_build.clicked.connect(self._build_table)

    def _refresh(self):
        pass

    def _build_table(self):
        opt = self.cb_opt.currentIndex() + 1
        res = self.state.analysis_results.get(opt)
        if res is None:
            QMessageBox.information(self, '알림', f'옵션 {opt} 분석을 먼저 실행하세요.')
            return

        df_valid  = res.get('df_valid', pd.DataFrame()).copy()
        corrected = res.get('corrected')
        z_scores  = res.get('z_scores')

        if corrected is not None:
            df_valid['보정값(µA)'] = corrected
        if z_scores is not None:
            df_valid['z_score']   = z_scores

        # 등급 필터
        grade_col = PROCESS_COL_GRADE
        filt = self.cb_grade.currentText()
        if filt == 'A만' and grade_col in df_valid.columns:
            df_valid = df_valid[df_valid[grade_col].astype(str).str.upper() == 'A']
        elif filt == 'E만' and grade_col in df_valid.columns:
            df_valid = df_valid[df_valid[grade_col].astype(str).str.upper() == 'E']

        # 표시 컬럼 순서 정리
        priority = [
            'tray_id', 'cell_no', 'layer', 'v_init', 't_final', 'delta_t',
            'rwiring', 'OCV1', 'OCV2', 'OCV3', 'OCV4', 'OCV7',
            f'i_{self.state.n_minutes}min', '보정값(µA)', 'z_score',
            PROCESS_COL_GRADE, 'Cell ID', 'Lot ID',
        ]
        cols = [c for c in priority if c in df_valid.columns]
        rest = [c for c in df_valid.columns if c not in cols]
        df_show = df_valid[cols + rest]

        self.table.setRowCount(len(df_show))
        self.table.setColumnCount(len(df_show.columns))
        self.table.setHorizontalHeaderLabels(list(df_show.columns))

        for r, (_, row) in enumerate(df_show.iterrows()):
            for c, val in enumerate(row):
                item = QTableWidgetItem(
                    f'{val:.4f}' if isinstance(val, float) else str(val)
                )
                # z_score 컬럼에서 불량 강조
                if df_show.columns[c] == 'z_score':
                    try:
                        if float(val) >= self.state.threshold:
                            item.setBackground(QColor(255, 180, 180))
                    except (ValueError, TypeError):
                        pass
                self.table.setItem(r, c, item)

        self.table.resizeColumnsToContents()


# ══════════════════════════════════════════════
#  Tab 7: 내보내기
# ══════════════════════════════════════════════

class Tab7Export(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('내보내기'))

        g_tbl = QGroupBox('결과 테이블 저장')
        tl = QVBoxLayout(g_tbl)
        self.btn_excel = QPushButton('Excel (.xlsx) 저장')
        self.btn_csv   = QPushButton('CSV 저장')
        tl.addWidget(self.btn_excel)
        tl.addWidget(self.btn_csv)
        layout.addWidget(g_tbl)

        g_img = QGroupBox('설정 저장/불러오기')
        il = QVBoxLayout(g_img)
        il.addWidget(QLabel('(향후 구현 예정)'))
        layout.addWidget(g_img)

        layout.addStretch()

        self.btn_excel.clicked.connect(self._save_excel)
        self.btn_csv.clicked.connect(self._save_csv)

    def _get_df(self) -> pd.DataFrame:
        frames = []
        for opt, res in self.state.analysis_results.items():
            df = res.get('df_valid', pd.DataFrame()).copy()
            c  = res.get('corrected')
            z  = res.get('z_scores')
            if c is not None:
                df[f'보정값_opt{opt}'] = c
            if z is not None:
                df[f'z_score_opt{opt}'] = z
            frames.append(df)
        if not frames:
            return self.state.df_meta.copy()
        # 전체를 한 df로 (첫 번째 기준 outer merge)
        merged = frames[0]
        for f in frames[1:]:
            new_cols = [c for c in f.columns if c not in merged.columns]
            if new_cols:
                merged = merged.merge(
                    f[['tray_id', 'cell_no'] + new_cols],
                    on=['tray_id', 'cell_no'], how='left',
                )
        return merged

    def _save_excel(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Excel 저장', 'SDM_result.xlsx',
            'Excel (*.xlsx)'
        )
        if path:
            try:
                self._get_df().to_excel(path, index=False)
                QMessageBox.information(self, '완료', f'저장 완료:\n{path}')
            except Exception as e:
                QMessageBox.critical(self, '오류', str(e))

    def _save_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'CSV 저장', 'SDM_result.csv',
            'CSV (*.csv)'
        )
        if path:
            try:
                self._get_df().to_csv(path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, '완료', f'저장 완료:\n{path}')
            except Exception as e:
                QMessageBox.critical(self, '오류', str(e))


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


# ══════════════════════════════════════════════
#  메인 윈도우
# ══════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SDM 분석 툴 — 리튬이차전지 자가방전 판정')
        self.resize(1280, 800)

        self.state = AppState()

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self.tab1 = Tab1Load(self.state)
        self.tab2 = Tab2Explore(self.state)
        self.tab3 = Tab3Heatmap(self.state)
        self.tab4 = Tab4Analysis(self.state)
        self.tab5 = Tab5Result(self.state)
        self.tab6 = Tab6Table(self.state)
        self.tab7 = Tab7Export(self.state)
        self.tab8 = Tab8Trend(self.state)

        tabs.addTab(self.tab1, '① 데이터 불러오기')
        tabs.addTab(self.tab2, '② 원시 데이터 탐색')
        tabs.addTab(self.tab3, '③ 히트맵')
        tabs.addTab(self.tab4, '④ 분석 로직')
        tabs.addTab(self.tab5, '⑤ 결과 및 판정')
        tabs.addTab(self.tab6, '⑥ 통합 결과 테이블')
        tabs.addTab(self.tab7, '⑦ 내보내기')
        tabs.addTab(self.tab8, '⑧ 다중 트레이 트렌드')

        # 분석 완료 → Tab5 자동 갱신, Tab8 세션 정보 갱신
        self.tab4.analysis_done.connect(lambda opt: self.tab5._draw())
        self.tab4.analysis_done.connect(lambda opt: self.tab8._refresh())


# ══════════════════════════════════════════════
#  진입점
# ══════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
