#!/usr/bin/env python3
"""Bar chart of pulls per time bucket, with a selectable period and step."""

import calendar
import datetime

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

# Dropdown label -> (period, step)
VIEWS = {
    "Day · 1 minute": ("day", "minute"),
    "Day · 1 hour": ("day", "hour"),
    "Week · 1 day": ("week", "day"),
    "Month · 1 week": ("month", "week"),
    "Month · 1 day": ("month", "day"),
    "Year · 1 month": ("year", "month"),
    "Year · 1 week": ("year", "week"),
    "Year · 1 day": ("year", "day"),
}

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def _period_bounds(period, ref):
    """Return the [start, end) datetimes of the period containing `ref`."""
    if period == "day":
        start = datetime.datetime.combine(ref, datetime.time.min)
        return start, start + datetime.timedelta(days=1)
    if period == "week":
        monday = ref - datetime.timedelta(days=ref.weekday())
        start = datetime.datetime.combine(monday, datetime.time.min)
        return start, start + datetime.timedelta(days=7)
    if period == "month":
        start = datetime.datetime(ref.year, ref.month, 1)
        if ref.month == 12:
            end = datetime.datetime(ref.year + 1, 1, 1)
        else:
            end = datetime.datetime(ref.year, ref.month + 1, 1)
        return start, end
    if period == "year":
        return (datetime.datetime(ref.year, 1, 1),
                datetime.datetime(ref.year + 1, 1, 1))
    raise ValueError(period)


def _bucket_starts(step, start, end):
    """Return the start datetime of every bucket in [start, end)."""
    starts = []
    if step == "month":
        year, month = start.year, start.month
        while datetime.datetime(year, month, 1) < end:
            starts.append(datetime.datetime(year, month, 1))
            month += 1
            if month > 12:
                month = 1
                year += 1
        return starts

    delta = {
        "minute": datetime.timedelta(minutes=1),
        "hour": datetime.timedelta(hours=1),
        "day": datetime.timedelta(days=1),
        "week": datetime.timedelta(days=7),
    }[step]

    current = start
    while current < end:
        starts.append(current)
        current += delta
    return starts


def _bucket_index(when, step, start):
    if step == "minute":
        return int((when - start).total_seconds() // 60)
    if step == "hour":
        return int((when - start).total_seconds() // 3600)
    if step == "day":
        return (when.date() - start.date()).days
    if step == "week":
        return (when.date() - start.date()).days // 7
    if step == "month":
        return (when.year - start.year) * 12 + (when.month - start.month)
    raise ValueError(step)


def build_graph_data(period, step, ref, pulls):
    """Return (counts, labels, ticks, title) for one graph view.

    counts  - pulls per bucket
    labels  - {bucket_index: axis label}
    ticks   - extra bucket indices that get a mark but no label
    title   - heading text
    """
    start, end = _period_bounds(period, ref)
    starts = _bucket_starts(step, start, end)
    counts = [0] * len(starts)

    for when in pulls:
        if start <= when < end:
            idx = _bucket_index(when, step, start)
            if 0 <= idx < len(counts):
                counts[idx] += 1

    labels = {}
    ticks = set()
    n = len(starts)

    if step == "minute":
        # Marks at every hour, plus a mark at any minute that has a pull.
        for i in range(0, n, 60):
            labels[i] = starts[i].strftime("%H:%M")
        ticks = {i for i, count in enumerate(counts) if count}

    elif step == "hour":
        for i in range(n):
            labels[i] = starts[i].strftime("%H:%M")

    elif step == "day" and period == "week":
        for i in range(n):
            labels[i] = WEEKDAYS[starts[i].weekday()]

    elif step == "day" and period == "month":
        for i in range(n):
            day = starts[i].day
            if day == 1 or day % 5 == 0:
                labels[i] = str(day)

    elif step == "day" and period == "year":
        for i in range(n):
            if starts[i].day == 1:
                labels[i] = calendar.month_abbr[starts[i].month]

    elif step == "week" and period == "month":
        last_day = (end - datetime.timedelta(days=1)).day
        for i in range(n):
            first = starts[i].day
            last = min(first + 6, last_day)
            labels[i] = f"{first}-{last}"

    elif step == "week" and period == "year":
        for i in range(0, n, 4):
            labels[i] = f"W{i + 1}"

    elif step == "month":
        for i in range(n):
            labels[i] = calendar.month_abbr[starts[i].month]

    if period == "day":
        when = start.strftime("%d %b %Y")
    elif period == "week":
        when = f"week of {start.strftime('%d %b %Y')}"
    elif period == "month":
        when = start.strftime("%B %Y")
    else:
        when = str(start.year)

    return counts, labels, ticks, f"Pulls per {step} · {when}"


class PullGraph(QWidget):
    """A simple bar chart with a y-axis count and a labelled x-axis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._counts = []
        self._labels = {}
        self._ticks = set()
        self._title = ""
        self.setMinimumHeight(230)

    def set_data(self, counts, labels, ticks, title):
        self._counts = counts
        self._labels = labels
        self._ticks = ticks
        self._title = title
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        left, right, top, bottom = 46, 12, 24, 34
        plot = QRectF(left, top,
                      max(1, self.width() - left - right),
                      max(1, self.height() - top - bottom))

        title_font = QFont()
        title_font.setPointSize(9)
        painter.setFont(title_font)
        painter.setPen(QColor(70, 70, 70))
        painter.drawText(
            QRectF(left, 2, self.width() - left - right, top - 6),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            self._title)

        if not self._counts or max(self._counts) == 0:
            painter.setPen(QColor(140, 140, 140))
            painter.drawText(plot, int(Qt.AlignmentFlag.AlignCenter),
                             "No pulls in this period")
            return

        y_max = max(self._counts)
        for candidate in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
                          2000, 5000, 10000):
            if candidate >= y_max:
                y_max = candidate
                break

        small_font = QFont()
        small_font.setPointSize(7)
        painter.setFont(small_font)

        y_ticks = [0, y_max]
        if y_max >= 2 and y_max % 2 == 0:
            y_ticks.insert(1, y_max // 2)

        for value in y_ticks:
            y = plot.bottom() - plot.height() * (value / y_max)
            painter.setPen(QPen(QColor(232, 232, 232), 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(90, 90, 90))
            painter.drawText(
                QRectF(0, y - 8, left - 6, 16),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                str(value))

        painter.setPen(QPen(QColor(160, 160, 160), 1))
        painter.drawLine(QPointF(plot.left(), plot.bottom()),
                         QPointF(plot.right(), plot.bottom()))
        painter.drawLine(QPointF(plot.left(), plot.bottom()),
                         QPointF(plot.left(), plot.top()))

        n = len(self._counts)
        bar_w = plot.width() / n
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(46, 125, 50))
        for i, count in enumerate(self._counts):
            if count <= 0:
                continue
            height = plot.height() * (count / y_max)
            x = plot.left() + i * bar_w
            width = max(1.0, bar_w * 0.9)
            painter.drawRect(QRectF(x, plot.bottom() - height, width, height))

        for i in sorted(set(self._labels) | self._ticks):
            x = plot.left() + (i + 0.5) * bar_w
            painter.setPen(QPen(QColor(160, 160, 160), 1))
            painter.drawLine(QPointF(x, plot.bottom()),
                             QPointF(x, plot.bottom() + 4))
            label = self._labels.get(i)
            if label:
                painter.setPen(QColor(70, 70, 70))
                painter.drawText(
                    QRectF(x - 32, plot.bottom() + 5, 64, 16),
                    int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
                    label)
