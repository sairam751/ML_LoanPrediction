#This file is for the exceptions
import sys

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in in python script name [{0}] line number [{1}] and error message [{2}]".format(
     file_name,exc_tb.tb_lineno,str(error)
     )
    
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message,error_detail=error_details)

    def __str__(self) -> str:
        return self.error_message