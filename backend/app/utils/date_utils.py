from datetime import datetime

def is_valid_date(date_string):
    """
    Checks if the string matches the expected format: DD-MM-YYYY (e.g., 02-03-2026).
    """
    try:
        datetime.strptime(date_string, '%d-%m-%Y')
        return True
    except (ValueError, TypeError):
        return False