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
