# ─────────────────────────────────────────────
#  constants.py  —  SDM 분석 툴 전역 상수
# ─────────────────────────────────────────────

# 설비번호 → 셀 번호 오프셋 (셀 1~144)
DEVICE_OFFSET = {5: 0, 4: 16, 3: 48, 2: 80, 1: 112}
DEVICE_CELL_RANGE = {
    5: range(1, 17),
    4: range(17, 49),
    3: range(49, 81),
    2: range(81, 113),
    1: range(113, 145),
}

TOTAL_CELLS = 144
TRAY_ROWS   = 12
TRAY_COLS   = 12
NUM_LAYERS  = 6   # 가장자리(1) → 중앙(6)

# ── 공정 데이터 컬럼명 ──────────────────────────
PROCESS_COL_TRAY_ID = 'TRAY ID_OCV #01'   # 대표 Tray ID 컬럼
PROCESS_COL_CELL_NO = 'CELL NO_OCV #01'   # 대표 Cell NO 컬럼

# 모델 독립변수 (OCV 시점)
PROCESS_COL_OCV = {
    'OCV1'        : 'OCV_OCV #01',
    'OCV2'        : 'OCV_OCV #02',
    'OCV3'        : 'OCV_OCV #03',
    'OCV4'        : 'OCV_OCV #04',
    'OCV7'        : 'OCV_OCV #07',
    'CHARGE_END_V': 'End Voltage_Charge #01',   # 1차 충전 종료 전압
}

# 검증용 dOCV (3일 기준)
PROCESS_COL_DOCV  = 'Delta OCV_Delta OCV #07'

# 판정 레이블 / 메타
PROCESS_COL_GRADE   = '판정등급'
PROCESS_COL_CELL_ID = 'Cell ID'
PROCESS_COL_LOT_ID  = 'Lot ID'

# ── 보정 기준 조건 ─────────────────────────────
REF_V_INIT  = 3590.0   # mV
REF_T_FINAL = 25.0     # °C
REF_DELTA_T = 0.0      # °C

# ── Robust 회귀 잔차 경계 ──────────────────────
ROBUST_LOW_PCT  = 10
ROBUST_HIGH_PCT = 90

# ── VIF 경고 기준 ──────────────────────────────
VIF_WARN = 10.0


def cell_to_label(cell_no: int) -> str:
    """셀 번호(1~144) → 위치 레이블 (A01~L12)"""
    row_letter = chr(ord('A') + (cell_no - 1) // 12)
    col_number = (cell_no - 1) % 12 + 1
    return f'{row_letter}{col_number:02d}'
