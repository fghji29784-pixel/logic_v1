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
