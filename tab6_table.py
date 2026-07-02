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
            'tray_id', 'cell_no', 'layer',
            'v_init', 'v_final', 'delta_v',
            't_init', 't_final', 'delta_t',
            'rwiring', 'OCV1', 'OCV2', 'OCV3', 'OCV4', 'OCV7',
            f'i_{self.state.n_minutes}min', '보정값(µA)', 'z_score',
            PROCESS_COL_DOCV,
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
