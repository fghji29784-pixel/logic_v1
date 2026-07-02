from __future__ import annotations
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False




# ══════════════════════════════════════════════
#  공통 Matplotlib 캔버스 위젯
# ══════════════════════════════════════════════

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, figsize=(6, 4)):
        self.fig = Figure(figsize=figsize, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

    def clear(self):
        self.fig.clear()
        self.draw()
