#!/usr/bin/env python3
"""Yearly activity heatmap: twelve month blocks in a row, with gaps."""

import calendar
import datetime

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

MONTH_GAP = 12
LABEL_WIDTH = 30
TOP = 22
MIN_CELL = 4
MAX_CELL = 16


def color_for(count):
    """Green at 0 pulls; every pull lowers G by 10 (green -> black)."""
    if count <= 0:
        return QColor(0, 255, 0)
    green = max(0, 255 - 10 * count)
    return QColor(0, green, 0)


class HeatmapWidget(QWidget):
    """12 month blocks (week columns x 7 weekdays) for a single year."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._year = datetime.date.today().year
        self._counts = {}
        self._cell_rects = []
        self._hovered = None
        self.setMouseTracking(True)
        self.setMinimumHeight(120)

    def set_data(self, year, counts):
        self._year = int(year)
        self._counts = counts or {}
        self._hovered = None
        self.update()

    def _week_columns(self, month):
        first_weekday, days = calendar.monthrange(self._year, month)
        return (first_weekday + days + 6) // 7

    def _layout_metrics(self):
        weeks = [self._week_columns(m) for m in range(1, 13)]
        total_weeks = sum(weeks)
        avail = self.width() - LABEL_WIDTH - MONTH_GAP * 11 - 8
        cell = int(avail / max(total_weeks, 1)) - 1
        cell = max(MIN_CELL, min(cell, MAX_CELL))
        return weeks, cell, cell + 1

    def resizeEvent(self, event):
        super().resizeEvent(event)
        _, _, step = self._layout_metrics()
        height = TOP + 7 * step + 6
        if self.height() != height:
            self.setFixedHeight(height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self._cell_rects = []

        months = list(range(1, 13))
        weeks, cell, step = self._layout_metrics()
        total_weeks = sum(weeks)

        used = LABEL_WIDTH + total_weeks * step + MONTH_GAP * 11
        x0 = max(4, (self.width() - used) // 2)

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        # Weekday labels (Monday-first), GitHub style: Mon/Wed/Fri only.
        painter.setPen(QColor(110, 110, 110))
        for row, name in ((0, "Mon"), (2, "Wed"), (4, "Fri")):
            painter.drawText(
                QRectF(x0, TOP + row * step, LABEL_WIDTH - 4, step),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                name)

        border = QPen(QColor(220, 220, 220), 1)
        x = x0 + LABEL_WIDTH

        for month, week_cols in zip(months, weeks):
            painter.setPen(QColor(80, 80, 80))
            painter.drawText(
                QRectF(x, 0, week_cols * step, TOP - 2),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                calendar.month_abbr[month])

            first_weekday, days = calendar.monthrange(self._year, month)
            for day in range(1, days + 1):
                index = first_weekday + day - 1
                col, row = divmod(index, 7)
                iso = datetime.date(self._year, month, day).isoformat()
                count = int(self._counts.get(iso, 0))
                rect = QRectF(x + col * step, TOP + row * step, cell, cell)
                painter.fillRect(rect, color_for(count))
                painter.setPen(border)
                painter.drawRect(rect)
                self._cell_rects.append(
                    (rect, datetime.date(self._year, month, day), count))

            x += week_cols * step + MONTH_GAP

    @staticmethod
    def _tooltip_text(day, count):
        if count == 0:
            amount = "No pulls"
        else:
            amount = f"{count} pull" + ("" if count == 1 else "s")
        return f"{day.strftime('%d %b %Y')}\n{amount}"

    def _cell_at(self, pos):
        for rect, day, count in self._cell_rects:
            if rect.contains(pos):
                return day, count
        return None

    def mouseMoveEvent(self, event):
        hit = self._cell_at(event.position())
        if hit is None:
            if self._hovered is not None:
                self._hovered = None
                QToolTip.hideText()
        else:
            day, count = hit
            if day != self._hovered:
                self._hovered = day
                QToolTip.showText(event.globalPosition().toPoint(),
                                  self._tooltip_text(day, count), self)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hovered = None
        QToolTip.hideText()
        super().leaveEvent(event)
