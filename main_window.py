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
from tab1_load import Tab1Load
from tab2_explore import Tab2Explore
from tab3_heatmap import Tab3Heatmap
from tab4_analysis import Tab4Analysis
from tab5_result import Tab5Result
from tab6_table import Tab6Table
from tab7_export import Tab7Export
from tab8_trend import Tab8Trend




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
        tabs.addTab(self.tab4.scatter_tab, '④-2 변수별 산점도')
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
