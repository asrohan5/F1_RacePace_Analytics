import sys
import logging

def get_error_details(error: Exception, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"
    return (f'Error in [{file_name}] at lineno [{line_number}]: {str(error)}')


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = get_error_details(error_message, error_detail)

    def __str__(self):
        return self.error_message