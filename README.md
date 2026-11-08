# COMP4330SEF-Project

## Project Members

| Name | Student ID |
|------|-----------|
| Lau Chun Kit | 13897100 |
| Choy Tsz Lok | 14108370 |
| Chong Tsz Ho | 13990354 |
| Tse Tsz Sum | 14231009 |
| Cheung Tsz Hin Anderson | 14113287 |
| Fung Ming Chu | 14201890 |

## Building Environment

1. Install **uv** by using the command in powershell

    ```bash
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2. Clone this repository

    ```bash
    git clone https://github.com/TsukiKazami/COMP4330SEF-Project
    ```

3. Type uv sync to build up the environment

    ```bash
    uv sync
    ```

4. Test whether the environment has been successfully built

    ```bash
    # The output should be: Ultralytics 8.4.207  Python-3.14.10 torch-2.15.0+cu132 CUDA:0 (GPU) Setup complete

    uv run yolo checks
    ```

    ```bash
    # The environment should be activated

    .venv\Scripts\activate
    ```

    ```bash
    # The output pictures should be shown on *\runs\detect\predict\bus.jpg

    yolo predict model=yolo26n.pt
    ```