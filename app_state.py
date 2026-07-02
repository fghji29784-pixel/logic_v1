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
