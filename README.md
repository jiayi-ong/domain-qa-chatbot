## Project Overview
This project implements a Q&A Chatbot that specializes in the metrics used in quantitative financial analysis. Using prompt and context engineering, the Chatbot is able to provide definitions of metrics such as the P/E ratio and EPS, and also explain their meanings, applications, limitations, and give examples. It is also instructed to handle out-of-scope and distressed user inputs.

## Running Evaluation

First, download the code repository. To run the model evaluation locally on Windows, run the following line of code in Command Prompt after replacing the placeholder values (be careful not to replace any quotation marks):
```
uv run --project "<FULL_PATH_TO_PROJECT_ROOT>" cmd /c "start /B python -m uvicorn domain_qa.app.main:app --reload && timeout /t <STARTUP_WAIT_SECONDS> >nul && python -m domain_qa.eval.run --base-url http://<HOST>:<PORT> --report-path <REL_PATH_TO_REPORT_FILE>"
```

```<FULL_PATH_TO_PROJECT_ROOT>```: The absolute path to the project root folder. E.g. ```C:\Desktop\cloned_project```

```<STARTUP_WAIT_SECONDS>```: The number of seconds (integer) to wait for app to start-up. E.g. ```10```

```<HOST>```: E.g. ```127.0.0.1```

```<PORT>```: E.g. ```8000```

```<REL_PATH_TO_REPORT_FILE>```: The relative path (relative to the current directory in Command Prompt) to export the text evaluation report (NOTE: has to end with .txt). E.g. ```reports\eval_report.txt```