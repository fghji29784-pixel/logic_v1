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

        # dOCV 대체재 검증
        g_sur = QGroupBox('dOCV 대체재 검증')
        g_sur.setToolTip('SDM 보정값이 기존 dOCV 판정을 얼마나 재현하는지.\n'
                         '불량 라벨이 거의 없어도 연속 dOCV 상관으로 매번 검증 가능.')
        sl = QVBoxLayout(g_sur)
        self.lbl_pear   = QLabel('Pearson r: —')
        self.lbl_spear  = QLabel('Spearman r: —')
        self.lbl_rnorm  = QLabel('양품만 r(P/S): —')
        self.lbl_cut    = QLabel('SDM 컷오프: —')
        self.lbl_agree  = QLabel('일치율(민감/특이): —')
        self.lbl_rnorm.setToolTip(
            '명백한 불량(dOCV E) 제외, 양품 범위 안에서의 상관.\n'
            '높으면 → SDM이 dOCV 대체 가능(미세불량 선별 가능).\n'
            '0에 가까우면 → SDM은 큰 단락만 잡는 검출기.')
        self.lbl_cut.setToolTip(
            'dOCV 규칙(트레이 median+0.8mV)에 대응하는 SDM 보정값 컷오프(µA).\n'
            'SDM↔dOCV 회귀로 역산. OCV1 시점 선별 기준선 후보.')
        self.lbl_agree.setToolTip(
            'SDM 컷오프로 dOCV 규칙 라벨을 재현했을 때\n민감도(불량검출률)/특이도(양품통과율).')
        for w in (self.lbl_pear, self.lbl_spear, self.lbl_rnorm,
                  self.lbl_cut, self.lbl_agree):
            sl.addWidget(w)
        _d_sur = QLabel('양품만 r 이 핵심 — 대체재 여부 판가름')
        _d_sur.setWordWrap(True)
        _d_sur.setStyleSheet('color:#777; font-size:10px;')
        sl.addWidget(_d_sur)
        cl.addWidget(g_sur)

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

    def _update_surrogate_labels(self, sur: dict | None):
        if not sur:
            for w, t in [(self.lbl_pear, 'Pearson r: —'),
                         (self.lbl_spear, 'Spearman r: —'),
                         (self.lbl_rnorm, '양품만 r(P/S): —'),
                         (self.lbl_cut, 'SDM 컷오프: —'),
                         (self.lbl_agree, '일치율(민감/특이): —')]:
                w.setText(t)
            return
        def f(v, fmt='{:.3f}'):
            return fmt.format(v) if isinstance(v, float) and not np.isnan(v) else '—'
        self.lbl_pear.setText(f"Pearson r: {f(sur.get('pearson'))}")
        self.lbl_spear.setText(f"Spearman r: {f(sur.get('spearman'))}")
        self.lbl_rnorm.setText(
            f"양품만 r(P/S): {f(sur.get('pearson_normal'))} / {f(sur.get('spearman_normal'))}")
        self.lbl_cut.setText(
            f"SDM 컷오프: {f(sur.get('sdm_cutoff'), '{:.2f}')} µA"
            if 'sdm_cutoff' in sur else 'SDM 컷오프: —')
        if 'sensitivity' in sur:
            self.lbl_agree.setText(
                f"일치율: 민감 {f(sur.get('sensitivity'))} / 특이 {f(sur.get('specificity'))} "
                f"(dOCV E={sur.get('n_docv_E','?')})")
        else:
            self.lbl_agree.setText(f"일치율: — (dOCV E={sur.get('n_docv_E','?')})")

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

        # ── dOCV 대체재 검증 (상관·컷오프 역산) ──
        sur = None
        if corrected is not None and docv_vals is not None:
            try:
                sur = docv_surrogate_analysis(df_valid, corrected, docv_col)
            except Exception:
                sur = None
        self._update_surrogate_labels(sur)
        # (SDM 컷오프 선은 실측 상관이 약해 오해 소지 → 산점도에 표시하지 않음.
        #  역산 컷오프 수치는 'dOCV 대체재 검증' 패널에만 참고용으로 유지.)

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
