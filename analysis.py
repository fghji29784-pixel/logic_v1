# ─────────────────────────────────────────────
#  analysis.py  —  SDM 판정 로직 (Steps 3~8)
# ─────────────────────────────────────────────

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

from constants import (
    NUM_LAYERS,
    PROCESS_COL_OCV, PROCESS_COL_GRADE, PROCESS_COL_DOCV,
    REF_V_INIT, REF_T_FINAL, REF_DELTA_T,
    ROBUST_LOW_PCT, ROBUST_HIGH_PCT, VIF_WARN,
)

warnings.filterwarnings('ignore', category=FutureWarning)


# ══════════════════════════════════════════════
#  Step 4: 레이어 더미 추가
# ══════════════════════════════════════════════

def add_layer_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """dummy_L2 ~ dummy_L6 컬럼 추가 (Layer 1 = reference)"""
    df = df.copy()
    for i in range(2, NUM_LAYERS + 1):
        df[f'dummy_L{i}'] = (df['layer'] == i).astype(float)
    return df


# ══════════════════════════════════════════════
#  Step 5~7: 독립변수 목록
# ══════════════════════════════════════════════

SDM_FEATURES     = ['v_init', 't_final', 'delta_t'] + [f'dummy_L{i}' for i in range(2, NUM_LAYERS + 1)]
PROCESS_FEATURES = list(PROCESS_COL_OCV.keys())   # OCV1~4, OCV7


def get_feature_cols(include_process: bool = False, df: pd.DataFrame | None = None) -> list[str]:
    """독립변수 목록 반환. df 전달 시 실제 존재하는 컬럼만 필터링."""
    cols = SDM_FEATURES + (PROCESS_FEATURES if include_process else [])
    if df is not None:
        cols = [c for c in cols if c in df.columns]
    return cols


# ══════════════════════════════════════════════
#  종속변수 준비
# ══════════════════════════════════════════════

def prepare_y(df: pd.DataFrame, n_minutes: int,
              dep_type: str = 'single') -> pd.Series:
    """
    종속변수(y) 시리즈 반환 (단위: µA 또는 µA/s).
    dep_type: 'single' = N분 단일값, 'slope' = 0~N분 기울기
    """
    if dep_type == 'single':
        col = f'i_{n_minutes}min'
    else:
        col = f'slope_0_{n_minutes}'

    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    return df[col] * 1e6   # A → µA (or A/s → µA/s)


# ══════════════════════════════════════════════
#  OCV 컬럼 이름 → 변수명 정규화
# ══════════════════════════════════════════════

def resolve_ocv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """공정 데이터 컬럼(OCV_OCV #01 등)을 OCV1 등 변수명으로 복사"""
    df = df.copy()
    for var, col in PROCESS_COL_OCV.items():
        if col in df.columns and var not in df.columns:
            df[var] = pd.to_numeric(df[col], errors='coerce')
    return df


# ══════════════════════════════════════════════
#  Step 3: 이상치 제거
# ══════════════════════════════════════════════

def remove_outliers(df: pd.DataFrame,
                    rwiring_threshold: float | None = None,
                    ocv_diff_threshold: float | None = None) -> pd.DataFrame:
    """
    Step 3: 이상치 제거
    - rwiring_threshold : Rwiring 초과 채널 제거 (Ω)
    - ocv_diff_threshold: 공정 OCV1 vs SDM V_init 괴리 기준 (mV)
    """
    mask = pd.Series(True, index=df.index)

    if rwiring_threshold is not None and 'rwiring' in df.columns:
        valid_rw = (df['rwiring'] <= rwiring_threshold) | df['rwiring'].isna()
        mask &= valid_rw

    if ocv_diff_threshold is not None and 'OCV1' in df.columns and 'v_init' in df.columns:
        ocv_mv = pd.to_numeric(df['OCV1'], errors='coerce') * 1000  # V→mV
        diff   = (df['v_init'] - ocv_mv).abs()
        mask  &= diff <= ocv_diff_threshold

    return df[mask].copy()


# ══════════════════════════════════════════════
#  회귀 모델 적합
# ══════════════════════════════════════════════

def fit_ols(X: pd.DataFrame, y: pd.Series):
    """OLS 회귀"""
    Xc = sm.add_constant(X, has_constant='add')
    return sm.OLS(y, Xc, missing='drop').fit()


def fit_robust(X: pd.DataFrame, y: pd.Series,
               low_pct: float = ROBUST_LOW_PCT,
               high_pct: float = ROBUST_HIGH_PCT) -> dict:
    """
    Robust 회귀 (3단계):
      1. OLS → 잔차
      2. 잔차 low~high percentile 내 재적합
      3. WLS (잔차² 역수 가중치)
    """
    Xc = sm.add_constant(X, has_constant='add')

    # 1단계 OLS
    ols1  = sm.OLS(y, Xc, missing='drop').fit()
    resid = ols1.resid

    # 2단계 잔차 필터
    lo, hi = np.percentile(resid, low_pct), np.percentile(resid, high_pct)
    mask2  = (resid >= lo) & (resid <= hi)
    ols2   = sm.OLS(y[mask2], Xc.loc[mask2], missing='drop').fit()
    resid2 = ols2.resid

    # 3단계 WLS
    weights = 1.0 / (resid2 ** 2 + 1e-12)
    wls     = sm.WLS(y[mask2], Xc.loc[mask2], weights=weights, missing='drop').fit()

    return {
        'model'       : wls,
        'normal_mask' : mask2,   # 정상 셀 마스크 (원본 y 인덱스 기준)
    }


def fit_lasso_loo_tray(X: pd.DataFrame, y: pd.Series,
                       groups: pd.Series) -> dict:
    """
    Lasso + LOO-tray CV
    groups: tray_id 시리즈 (최소 2개 트레이 필요)
    """
    valid = X.notna().all(axis=1) & y.notna()
    Xv, yv, gv = X[valid], y[valid], groups[valid]

    n_trays = len(set(gv.values))
    if n_trays < 2:
        raise ValueError(
            f'Lasso LOO-tray CV는 최소 2개 트레이가 필요합니다. (현재 {n_trays}개)'
        )

    scaler    = StandardScaler()
    Xs        = scaler.fit_transform(Xv)
    # sklearn은 array-like 기대 → .values 사용
    cv_splits = list(LeaveOneGroupOut().split(Xs, yv.values, groups=gv.values))

    lasso = LassoCV(cv=cv_splits, max_iter=20000, n_jobs=-1)
    lasso.fit(Xs, yv.values)

    return {
        'model'  : lasso,
        'scaler' : scaler,
        'alpha'  : lasso.alpha_,
        'coef'   : dict(zip(Xv.columns, lasso.coef_)),
        'valid'  : valid,
    }


# ══════════════════════════════════════════════
#  VIF 계산
# ══════════════════════════════════════════════

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """분산팽창계수(VIF) 반환. 10 초과 시 경고 플래그."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xc  = sm.add_constant(X.dropna(), has_constant='add')
    vif = pd.DataFrame({
        'feature': X.columns,
        'VIF'    : [variance_inflation_factor(Xc.values, i + 1) for i in range(len(X.columns))],
    })
    vif['warn'] = vif['VIF'] > VIF_WARN
    return vif


# ══════════════════════════════════════════════
#  Step 6: 보정값 계산
# ══════════════════════════════════════════════

def correct_values(y: pd.Series,
                   X: pd.DataFrame,
                   params: pd.Series,
                   ref: dict | None = None) -> pd.Series:
    """
    보정식:
      I_corrected = I_measured - Σ β_k × (x_k - x_k_ref)

    ref: 기준 조건 딕셔너리. 없으면 CLAUDE.md 기준값 사용.
    """
    default_ref = {
        'v_init' : REF_V_INIT,
        't_final': REF_T_FINAL,
        'delta_t': REF_DELTA_T,
        **{f'dummy_L{i}': 0.0 for i in range(2, NUM_LAYERS + 1)},
        **{f'OCV{i}': 0.0 for i in [1, 2, 3, 4, 7]},
    }
    if ref:
        default_ref.update(ref)

    correction = pd.Series(0.0, index=y.index)
    for feat in X.columns:
        if feat in params.index and feat in X.columns:
            ref_val = default_ref.get(feat, 0.0)
            correction += params[feat] * (X[feat] - ref_val)

    return y - correction


# ══════════════════════════════════════════════
#  Step 7: 표준화
# ══════════════════════════════════════════════

def standardize(corrected: pd.Series,
                normal_mask: pd.Series | None = None) -> pd.Series:
    """
    Z-score 표준화.
    normal_mask: 정상 셀 집단 마스크 (불량 오염 방지)
    """
    if normal_mask is not None and normal_mask.any():
        base = corrected[normal_mask]
    else:
        base = corrected
    mu    = base.mean()
    sigma = base.std(ddof=1)
    return (corrected - mu) / (sigma + 1e-10)


# ══════════════════════════════════════════════
#  Step 8: 평가지표
# ══════════════════════════════════════════════

def compute_metrics(z_scores: pd.Series,
                    true_labels: pd.Series | None) -> dict:
    """
    평가지표:
      d_prime = (µ_bad - µ_good) / σ_good
      AUC
      혼동행렬 (TP/FP/FN/TN) at z=2 기본 기준선
    """
    metrics = {}
    if true_labels is None or true_labels.isna().all():
        return metrics

    binary = (true_labels.astype(str).str.upper() == 'E').astype(int)
    valid  = ~(z_scores.isna() | binary.isna())
    z, b   = z_scores[valid], binary[valid]

    if b.sum() == 0:
        metrics['note'] = '불량셀(E) 없음'
        return metrics

    good_mask = b == 0
    bad_mask  = b == 1

    mu_g  = z[good_mask].mean()
    mu_b  = z[bad_mask].mean()
    sig_g = z[good_mask].std(ddof=1) if good_mask.sum() > 1 else 1.0

    metrics['d_prime']   = float((mu_b - mu_g) / (sig_g + 1e-10))
    metrics['mu_good']   = float(mu_g)
    metrics['mu_bad']    = float(mu_b)
    metrics['sigma_good']= float(sig_g)
    metrics['n_good']    = int(good_mask.sum())
    metrics['n_bad']     = int(bad_mask.sum())

    try:
        from sklearn.metrics import roc_auc_score
        metrics['auc'] = float(roc_auc_score(b, z))
    except Exception:
        metrics['auc'] = np.nan

    return metrics


def confusion_at_threshold(z_scores: pd.Series,
                            true_labels: pd.Series,
                            threshold: float) -> dict:
    """주어진 z-score 기준선에서 TP/FP/FN/TN 계산"""
    binary = (true_labels.astype(str).str.upper() == 'E').astype(int)
    pred   = (z_scores >= threshold).astype(int)
    valid  = ~(z_scores.isna() | binary.isna())
    b, p   = binary[valid], pred[valid]

    tp = int(((p == 1) & (b == 1)).sum())
    fp = int(((p == 1) & (b == 0)).sum())
    fn = int(((p == 0) & (b == 1)).sum())
    tn = int(((p == 0) & (b == 0)).sum())
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


# ══════════════════════════════════════════════
#  분리도 N분 커브
# ══════════════════════════════════════════════

def separation_curve(df_meta: pd.DataFrame,
                     option: int,
                     dep_type: str = 'single',
                     n_range: range | None = None,
                     rwiring_threshold: float | None = None,
                     ref: dict | None = None) -> pd.DataFrame:
    """
    N분을 5~15 구간에서 바꿔가며 d_prime 계산 → 최단 판정 시간 탐색용.
    반환: DataFrame [n_minutes, d_prime]
    """
    if n_range is None:
        n_range = range(5, 16)

    records = []
    for n in n_range:
        try:
            res = run_analysis(df_meta, option=option, n_minutes=n,
                               dep_type=dep_type,
                               rwiring_threshold=rwiring_threshold,
                               ref_conditions=ref)
            dp = res['metrics'].get('d_prime', np.nan)
        except Exception:
            dp = np.nan
        records.append({'n_minutes': n, 'd_prime': dp})

    return pd.DataFrame(records)


# ══════════════════════════════════════════════
#  전체 파이프라인 (Steps 3~8)
# ══════════════════════════════════════════════

def run_analysis(df_meta: pd.DataFrame,
                 option: int,
                 n_minutes: int,
                 dep_type: str = 'single',
                 rwiring_threshold: float | None = None,
                 ref_conditions: dict | None = None) -> dict:
    """
    옵션 1~5 분석 파이프라인.

    반환 dict 키:
      option, feature_cols, model(또는 lasso),
      vif, corrected, z_scores, df_valid, metrics
    """
    # OCV 컬럼 이름 정규화
    df = resolve_ocv_columns(df_meta)

    # 레이어 더미 추가
    df = add_layer_dummies(df)

    # 종속변수
    y_all = prepare_y(df, n_minutes, dep_type)
    df['y'] = y_all

    # Step 3: 이상치 제거
    df = remove_outliers(df, rwiring_threshold=rwiring_threshold)

    # 독립변수 목록
    include_process = option in [2, 4, 5]
    feature_cols    = get_feature_cols(include_process=include_process, df=df)

    y = df['y']
    X = df[feature_cols]

    # 결측 제거 후 유효 인덱스
    valid = X.notna().all(axis=1) & y.notna()
    Xv, yv, dfv = X[valid], y[valid], df[valid]

    out: dict = {
        'option'      : option,
        'n_minutes'   : n_minutes,
        'dep_type'    : dep_type,
        'feature_cols': feature_cols,
        'df_valid'    : dfv,
    }

    # ── 모델 적합 ──────────────────────────────
    normal_mask = pd.Series(True, index=dfv.index)   # 기본: 전체 정상

    if option in [1, 2]:
        model  = fit_ols(Xv, yv)
        params = model.params.drop('const', errors='ignore')
        out['model'] = model

    elif option in [3, 4]:
        rob    = fit_robust(Xv, yv)
        model  = rob['model']
        nm     = rob['normal_mask']           # yv 인덱스 기준 bool
        normal_mask = nm
        params = model.params.drop('const', errors='ignore')
        out['model']       = model
        out['normal_mask'] = normal_mask

    elif option == 5:
        lasso  = fit_lasso_loo_tray(Xv, yv, groups=dfv['tray_id'])
        out['lasso'] = lasso
        model  = None
        # 표준화 공간 계수 → 원 단위 계수로 변환: β_orig = β_std / scale
        scaler = lasso['scaler']
        params = pd.Series(
            lasso['model'].coef_ / (scaler.scale_ + 1e-10),
            index=feature_cols
        )

    else:
        raise ValueError(f'지원하지 않는 옵션: {option}')

    # ── VIF ────────────────────────────────────
    try:
        out['vif'] = compute_vif(Xv)
    except Exception:
        out['vif'] = None

    # ── Step 6: 보정값 ─────────────────────────
    corrected = correct_values(yv, Xv, params, ref=ref_conditions)
    out['corrected'] = corrected

    # ── Step 7: 표준화 ─────────────────────────
    if option in [3, 4]:
        z_scores = standardize(corrected, normal_mask=normal_mask)
    else:
        z_scores = standardize(corrected)
    out['z_scores'] = z_scores

    # ── Step 8: 평가지표 ───────────────────────
    grade_col   = PROCESS_COL_GRADE
    true_labels = dfv[grade_col] if grade_col in dfv.columns else None
    out['metrics'] = compute_metrics(z_scores, true_labels)

    return out
