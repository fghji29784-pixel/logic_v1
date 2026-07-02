# ─────────────────────────────────────────────
#  main.py  —  SDM 분석 툴 진입점 (실제 구현은 각 모듈로 분리)
#  실행: python main.py
#  배포: pyinstaller --onefile --windowed main.py
# ─────────────────────────────────────────────
from __future__ import annotations
from main_window import main

if __name__ == '__main__':
    main()
