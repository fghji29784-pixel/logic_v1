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

        # ── 오른쪽: 결과 출력 (스크롤) ──
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

        # 변수별 산점도 → 별도 탭('④-2 변수별 산점도')에 크게 표시
        _d4hint = QLabel('▶ 변수별 산점도는 "④-2 변수별 산점도" 탭에서 크게 확인하세요.')
        _d4hint.setWordWrap(True)
        _d4hint.setStyleSheet('color:#0066aa; font-size:10px;')
        rl2.addWidget(_d4hint)
        self._build_scatter_tab()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(right)
        layout.addWidget(scroll)

        # 시그널
        self.btn_run_one.clicked.connect(self._run_one)
        self.btn_run_all.clicked.connect(self._run_all)
        self.cb_tray.currentIndexChanged.connect(self._on_tray_changed)
        self.cb_tray_scatter.currentIndexChanged.connect(self._on_tray_changed)

    def _on_tray_changed(self, idx):
        """④ / ④-2 두 트레이 드롭다운을 동기화한 뒤 화면 갱신."""
        for cb in (self.cb_tray, self.cb_tray_scatter):
            if cb.currentIndex() != idx:
                cb.blockSignals(True)
                cb.setCurrentIndex(idx)
                cb.blockSignals(False)
        self._update_tray_display()

    def _build_scatter_tab(self):
        """변수별 산점도를 담는 별도 탭 위젯 구성.
        캔버스는 이 탭에 배치하되, Tab4의 _draw_scatter 가 그린다."""
        self.scatter_tab = QWidget()
        sl = QVBoxLayout(self.scatter_tab)

        # 트레이 선택 (④ 분석 로직 탭의 드롭다운과 동기화)
        tray_row = QHBoxLayout()
        tray_row.addWidget(QLabel('트레이 선택:'))
        self.cb_tray_scatter = QComboBox()
        self.cb_tray_scatter.addItem('전체 (합산)')
        self.cb_tray_scatter.setToolTip('④ 분석 로직 탭의 트레이 선택과 동기화됩니다.')
        tray_row.addWidget(self.cb_tray_scatter)
        tray_row.addStretch()
        sl.addLayout(tray_row)

        _d4 = QLabel('변수별 산점도: 각 독립변수 vs SDM 측정값(y, 보정 전). '
                     '빨간 회귀선 기울기 = 그 변수의 단독 영향. '
                     '점이 선을 따라 모일수록 그 변수가 SDM과 강한 상관 → 보정 효과 큼.  '
                     '(파랑=양품A, 빨강=불량E, 회색=미분류)')
        _d4.setWordWrap(True)
        _d4.setStyleSheet('color:#777; font-size:11px;')
        sl.addWidget(_d4)

        self.canvas_scatter = PlotCanvas(figsize=(11, 8))
        self.canvas_scatter.setMinimumSize(720, 560)
        sl.addWidget(NavigationToolbar2QT(self.canvas_scatter, self.scatter_tab))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas_scatter)
        sl.addWidget(scroll)

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
        self.state.selected_option = res.get('option', self.state.selected_option)
        # 트레이 목록 갱신 (선택 유지) — ④ / ④-2 두 드롭다운 동일하게
        prev  = self.cb_tray.currentText()
        items = ['전체 (합산)'] + [str(tid) for tid in res.get('per_tray_results', {})]
        for cb in (self.cb_tray, self.cb_tray_scatter):
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(items)
            idx = cb.findText(prev)
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            cb.blockSignals(False)
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
        if corrected is not None:
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

        # ── 변수별 산점도 (변수 vs SDM 측정값 y) ──
        self._draw_scatter(res, df_valid, title_suffix)

    def _draw_scatter(self, res: dict, df_valid: pd.DataFrame, title_suffix: str):
        """각 독립변수 vs SDM 측정값(y) 산점도 + 단독 회귀선."""
        feature_cols = res.get('feature_cols', [])
        self.canvas_scatter.fig.clear()
        if not feature_cols or 'y' not in df_valid.columns or df_valid.empty:
            self.canvas_scatter.draw()
            return

        y = pd.to_numeric(df_valid['y'], errors='coerce')

        # 등급 마스크
        if PROCESS_COL_GRADE in df_valid.columns:
            grades = df_valid[PROCESS_COL_GRADE].astype(str).str.strip().str.upper()
        else:
            grades = pd.Series('', index=df_valid.index)

        n_feat = len(feature_cols)
        ncol   = 3
        nrow   = (n_feat + ncol - 1) // ncol
        axes   = self.canvas_scatter.fig.subplots(nrow, ncol, squeeze=False)

        for k, feat in enumerate(feature_cols):
            ax = axes[k // ncol][k % ncol]
            if feat not in df_valid.columns:
                ax.set_visible(False)
                continue
            x = pd.to_numeric(df_valid[feat], errors='coerce')
            m = x.notna() & y.notna()
            if m.sum() == 0:
                ax.set_visible(False)
                continue
            xm, ym, gm = x[m], y[m], grades[m]

            for lbl, col, a, s in [('', '#888', 0.4, 8),
                                    ('A', 'steelblue', 0.5, 8),
                                    ('E', 'red', 0.9, 16)]:
                sel = (gm == lbl) if lbl else ~gm.isin(['A', 'E'])
                if sel.any():
                    ax.scatter(xm[sel], ym[sel], c=col, alpha=a, s=s,
                               zorder=3 if lbl == 'E' else 1)

            # 단독 회귀선 + 상관계수
            if xm.nunique() > 1:
                b, a0 = np.polyfit(xm, ym, 1)
                xs = np.array([xm.min(), xm.max()])
                ax.plot(xs, a0 + b * xs, color='red', lw=1.2, alpha=0.8)
                r = np.corrcoef(xm, ym)[0, 1]
                ax.set_title(f'{feat}  (r={r:.2f})', fontsize=8)
            else:
                ax.set_title(feat, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        # 남는 subplot 숨김
        for k in range(n_feat, nrow * ncol):
            axes[k // ncol][k % ncol].set_visible(False)

        self.canvas_scatter.fig.suptitle(
            f'변수별 산점도 [{title_suffix}]  (y=SDM 측정값 µA)', fontsize=9)
        self.canvas_scatter.fig.tight_layout()
        self.canvas_scatter.draw()
