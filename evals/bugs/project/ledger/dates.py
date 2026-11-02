from datetime import date, timedelta


def add_business_days(start: date, days: int) -> date:
    """Move forward `days` business days (Mon-Fri). 0 returns start (rolled to Monday if weekend)."""
    current = start
    while current.weekday() >= 5:
        current += timedelta(days=1)
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def due_date(issued: date, net_days: int) -> date:
    """Calendar net terms; if the due date lands on a weekend it moves to the next Monday."""
    due = issued + timedelta(days=net_days)
    while due.weekday() >= 5:
        due += timedelta(days=1)
    return due


def days_overdue(due: date, today: date) -> int:
    return max(0, (today - due).days)
