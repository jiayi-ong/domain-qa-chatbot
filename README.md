## Project Overview
This project implements a Q&A Chatbot that specializes in the metrics used in quantitative financial analysis. Using prompt and context engineering, the Chatbot is able to provide definitions of metrics such as the P/E ratio and EPS, and also explain their meanings, applications, limitations, and give examples. It is also instructed to handle out-of-scope and distressed user inputs.

## Running Evaluation

First, download the code repository. To run the model evaluation locally on Windows, run the following line of code in Command Prompt after replacing the placeholder values:
```
uv run --project "<FULL_PATH_TO_PROJECT_ROOT>" cmd /c "start /B python -m uvicorn domain_qa.app.main:app --reload && timeout /t <STARTUP_WAIT_SECONDS> >nul && python -m domain_qa.eval.run --base-url http://<HOST>:<PORT> --report-path <RELATIVE_PATH_TO_REPORT_FILE>"
```

```<FULL_PATH_TO_PROJECT_ROOT>```: The absolute path to the project root.

```<STARTUP_WAIT_SECONDS>```: The number of seconds to wait for app to start-up. Recommend 10 seconds.

```<HOST>```: e.g. ```127.0.0.1```

```<PORT>```: e.g. ```8000```

```<RELATIVE_PATH_TO_REPORT_FILE>```: The relative path (to root directory) to store evaluation reports.