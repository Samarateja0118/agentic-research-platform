from datetime import date

from ledger.dates import add_business_days, days_overdue, due_date


def test_business_days_skip_weekend():
    assert add_business_days(date(2026, 1, 2), 1) == date(2026, 1, 5)  # Fri + 1 -> Mon


def test_business_days_zero_on_weekend_rolls_forward():
    assert add_business_days(date(2026, 1, 3), 0) == date(2026, 1, 5)


def test_business_days_multiple():
    assert add_business_days(date(2026, 1, 5), 5) == date(2026, 1, 12)


def test_due_date_weekday():
    assert due_date(date(2026, 1, 5), 10) == date(2026, 1, 15)


def test_due_date_weekend_moves_to_monday():
    assert due_date(date(2026, 1, 1), 2) == date(2026, 1, 5)  # Sat -> Mon


def test_days_overdue():
    assert days_overdue(date(2026, 1, 10), date(2026, 1, 15)) == 5


def test_not_overdue_is_zero():
    assert days_overdue(date(2026, 1, 10), date(2026, 1, 1)) == 0
