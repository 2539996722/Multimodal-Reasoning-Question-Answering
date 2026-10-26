@echo off
rem ==========================================================================
rem  InduDoc AI Windows 启动脚本
rem  子命令：start（后台启动 + PID 文件）| stop | status | restart | run（前台，默认）
rem  端口：默认 8000，可用环境变量 PORT / HOST 覆盖，与 runtime/browser-check.mjs 保持一致
rem ==========================================================================
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul 2>nul
cd /d "%~dp0"

set "APP_HOME=%CD%"
set "RUNTIME_DIR=%APP_HOME%\runtime"
set "PID_FILE=%RUNTIME_DIR%\server.pid"
set "LOG_FILE=%RUNTIME_DIR%\server.log"
set "ERR_LOG_FILE=%RUNTIME_DIR%\server.error.log"

if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8000"
set "HOST=%HOST: =%"
set "PORT=%PORT: =%"

set "ACTION=%~1"
if not defined ACTION set "ACTION=run"

if /i "%ACTION%"=="run"     goto :dispatch
if /i "%ACTION%"=="start"   goto :dispatch
if /i "%ACTION%"=="stop"    goto :do_stop
if /i "%ACTION%"=="status"  goto :do_status_pre
if /i "%ACTION%"=="restart" goto :dispatch

echo [错误] 未知子命令：%ACTION%
echo 可用子命令：start ^| stop ^| status ^| restart（不带参数等同于 run，前台运行）
exit /b 2

rem --------------------------------------------------------------------------
rem  Python 探测：项目 .venv -> 已知 Conda 路径 -> py -3 -> PATH python
rem --------------------------------------------------------------------------
:dispatch
call :detect_python
if not defined APP_PYTHON (
    if defined PY_FOUND goto :deps_missing
    echo [错误] 未检测到可用的 Python 解释器。
    echo 请按下列步骤处理后重试：
    echo   1^) 创建项目虚拟环境：py -3 -m venv .venv
    echo   2^) 安装依赖：.venv\Scripts\python.exe -m pip install -r requirements.txt
    echo   3^) 或安装 Python 3.10+ 并确保 python 或 py -3 可用
    exit /b 1
)

if /i "%ACTION%"=="run"     goto :do_run
if /i "%ACTION%"=="start"   goto :do_start
if /i "%ACTION%"=="restart" goto :do_restart
exit /b 2

rem --------------------------------------------------------------------------
rem  前台运行（向后兼容：直接 .\start.bat 即进入前台）
rem --------------------------------------------------------------------------
:do_run
call :check_not_running
if "%START_BLOCKED%"=="1" exit /b 0
if "%START_BLOCKED%"=="2" exit /b 1
echo [InduDoc AI] Python : %APP_PYTHON% %APP_PYTHON_ARGS%
echo [InduDoc AI] 地址   : http://%HOST%:%PORT%
echo [InduDoc AI] 模式   : 前台运行（Ctrl+C 停止）
echo [InduDoc AI] 提示   : 后台运行请使用 start.bat start
"%APP_PYTHON%" %APP_PYTHON_ARGS% -m uvicorn app.main:app --host "%HOST%" --port "%PORT%"
set "RUN_EXIT=%ERRORLEVEL%"
if not "%RUN_EXIT%"=="0" echo [错误] 服务已退出，退出码 %RUN_EXIT%。
exit /b %RUN_EXIT%

rem --------------------------------------------------------------------------
rem  后台启动
rem --------------------------------------------------------------------------
:do_start
if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%" >nul 2>nul

call :check_not_running
if "%START_BLOCKED%"=="1" exit /b 0
if "%START_BLOCKED%"=="2" exit /b 1

echo [InduDoc AI] Python : %APP_PYTHON% %APP_PYTHON_ARGS%
echo [InduDoc AI] 地址   : http://%HOST%:%PORT%
echo [InduDoc AI] 正在后台启动 ...

set "UVICORN_ARGS=%APP_PYTHON_ARGS% -m uvicorn app.main:app --host %HOST% --port %PORT%"
set "PYTHONIOENCODING=utf-8"
rem 由 PowerShell 直接把 PID 写入 PID 文件：避免 for /f 捕获管道被子进程继承导致命令永不返回
del /q "%PID_FILE%" >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $q=[char]34; $exe='%APP_PYTHON%'; $log='%LOG_FILE%'; $errlog='%ERR_LOG_FILE%'; $argList=@(); foreach ($a in ('%UVICORN_ARGS%' -split '\s+')) { if ($a) { $argList += $a } }; $spid=0; $created=$null; try { $cmdLine='cmd /c ' + $q + $q + $exe + $q + ' ' + ($argList -join ' ') + ' > ' + $q + $log + $q + ' 2> ' + $q + $errlog + $q + $q; $created=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine=$cmdLine; CurrentDirectory='%APP_HOME%' } } catch { $created=$null }; if ($created -and $created.ReturnValue -eq 0) { $spid=$created.ProcessId; for ($i=0; $i -lt 40; $i++) { $child=Get-CimInstance Win32_Process -Filter ('ParentProcessId=' + $spid) | Where-Object { $_.Name -like 'python*' } | Select-Object -First 1; if ($child) { $spid=$child.ProcessId; break }; Start-Sleep -Milliseconds 200 } } else { $p=Start-Process -FilePath $exe -ArgumentList $argList -WorkingDirectory '%APP_HOME%' -RedirectStandardOutput $log -RedirectStandardError $errlog -WindowStyle Hidden -PassThru; $spid=$p.Id }; $spid | Out-File -Encoding ascii -FilePath '%PID_FILE%'" >nul 2>&1
if errorlevel 1 (
    echo [错误] 后台启动失败：无法创建服务进程。
    echo 请查看错误日志：%ERR_LOG_FILE%
    exit /b 1
)

call :read_pid
if not defined TARGET_PID (
    echo [错误] 后台启动失败：未能读取服务进程 PID（%PID_FILE%）。
    echo 请查看错误日志：%ERR_LOG_FILE%
    exit /b 1
)

rem 等待端口就绪，确认进程真实存活（不使用延迟展开，避免路径中的 ! 被吞）
set /a WAIT_COUNT=0
:wait_for_port
call :detect_port_pid
if defined PORT_PID goto :start_ok
call :is_alive "%TARGET_PID%"
if "%IS_ALIVE%"=="0" goto :start_failed
set /a WAIT_COUNT+=1
if %WAIT_COUNT% GEQ 30 goto :start_failed
ping -n 2 127.0.0.1 >nul 2>nul
goto :wait_for_port

:start_ok
echo [InduDoc AI] 服务已在后台启动。
echo   PID      : %TARGET_PID%
echo   地址     : http://%HOST%:%PORT%
echo   PID 文件 : %PID_FILE%
echo   日志     : %LOG_FILE%
echo   错误日志 : %ERR_LOG_FILE%
echo   停止服务 : start.bat stop
exit /b 0

:start_failed
call :is_alive "%TARGET_PID%"
if "%IS_ALIVE%"=="0" (
    echo [错误] 启动后进程已退出，请查看错误日志：%ERR_LOG_FILE%
    del /q "%PID_FILE%" >nul 2>nul
    exit /b 1
)
echo [错误] 进程已启动但端口 %PORT% 未在预期时间内就绪，请查看错误日志：%ERR_LOG_FILE%
exit /b 1

rem --------------------------------------------------------------------------
rem  停止（只停止本脚本启动的进程）
rem --------------------------------------------------------------------------
:do_stop
call :read_pid
if not defined TARGET_PID (
    echo [提示] 未找到 PID 文件 %PID_FILE%，当前没有由本脚本启动的服务。
    exit /b 0
)
call :is_alive "%TARGET_PID%"
if "%IS_ALIVE%"=="0" (
    echo [提示] 进程 %TARGET_PID% 已不存在，已清理 PID 文件。
    del /q "%PID_FILE%" >nul 2>nul
    exit /b 0
)
taskkill /PID %TARGET_PID% /T /F >nul 2>nul
if errorlevel 1 (
    echo [错误] 停止进程 %TARGET_PID% 失败，请手动执行：taskkill /PID %TARGET_PID% /T /F
    exit /b 1
)
del /q "%PID_FILE%" >nul 2>nul
echo [InduDoc AI] 已停止本脚本启动的服务（PID %TARGET_PID%）。
exit /b 0

rem --------------------------------------------------------------------------
rem  状态
rem --------------------------------------------------------------------------
:do_status_pre
call :detect_python
goto :do_status

:do_status
call :read_pid
call :detect_port_pid
set "RUNNING=0"
set "IS_ALIVE=0"
if defined TARGET_PID call :is_alive "%TARGET_PID%"
if "%IS_ALIVE%"=="1" set "RUNNING=1"
echo [InduDoc AI] 服务状态
echo   Python   : %APP_PYTHON% %APP_PYTHON_ARGS%
echo   地址     : http://%HOST%:%PORT%
echo   端口     : %PORT%
echo   主机     : %HOST%
if defined TARGET_PID (echo   PID 文件 : %TARGET_PID% ^(%PID_FILE%^)) else (echo   PID 文件 : 无 ^(%PID_FILE%^))
if "%RUNNING%"=="1" (
    echo   运行状态 : 运行中（由本脚本启动）
) else (
    echo   运行状态 : 未运行
)
if defined PORT_PID echo   端口占用 : 是（PID %PORT_PID%）
if not defined PORT_PID echo   端口占用 : 否
set "IS_ALIVE=0"
if defined PORT_PID call :is_alive "%PORT_PID%"
if defined PORT_PID if "%IS_ALIVE%"=="0" echo   提示     : 端口占用进程已不存在，可能是残留状态。
if defined PORT_PID if "%IS_ALIVE%"=="1" if "%PORT_PID%"=="%TARGET_PID%" echo   提示     : 端口由本脚本启动的服务占用。
if defined PORT_PID if "%IS_ALIVE%"=="1" if not "%PORT_PID%"=="%TARGET_PID%" if "%RUNNING%"=="1" echo   提示     : 端口由本脚本启动的服务占用（PID 文件记录的是启动器进程）。
if defined PORT_PID if "%IS_ALIVE%"=="1" if not "%PORT_PID%"=="%TARGET_PID%" if "%RUNNING%"=="0" echo   提示     : 该进程不是由本脚本启动，start.bat stop 不会停止它。
if "%RUNNING%"=="1" exit /b 0
exit /b 3

rem --------------------------------------------------------------------------
rem  重启
rem --------------------------------------------------------------------------
:do_restart
call :do_stop
if errorlevel 1 exit /b 1
goto :do_start

rem ==========================================================================
rem  子过程
rem ==========================================================================

:detect_python
set "APP_PYTHON="
set "APP_PYTHON_ARGS="
set "PY_FOUND="
set "PY_FIRST="
rem 1^) 项目虚拟环境
if exist "%APP_HOME%\.venv\Scripts\python.exe" call :try_python "%APP_HOME%\.venv\Scripts\python.exe" ""
if defined APP_PYTHON goto :eof
rem 2^) 已知 Conda 路径
for %%P in (
    "%CONDA_PREFIX%\python.exe"
    "%CONDA_PREFIX%\envs\qc\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%USERPROFILE%\anaconda3\envs\qc\python.exe"
    "C:\ProgramData\anaconda3\python.exe"
    "C:\ProgramData\miniconda3\python.exe"
    "C:\ProgramData\anaconda3\envs\qc\python.exe"
    "D:\anaconda3\python.exe"
    "D:\anaconda\python.exe"
    "D:\anaconda\envs\qc\python.exe"
    "D:\anaconda3\envs\qc\python.exe"
) do (
    if not defined APP_PYTHON call :try_python "%%~P" ""
)
if defined APP_PYTHON goto :eof
rem 3^) py -3
where py >nul 2>nul
if not errorlevel 1 (
    for /f "usebackq delims=" %%P in (`where py 2^>nul`) do (
        if not defined APP_PYTHON call :try_python "%%P" "-3"
    )
)
if defined APP_PYTHON goto :eof
rem 4^) PATH 中的 python
where python >nul 2>nul
if not errorlevel 1 (
    for /f "usebackq delims=" %%P in (`where python 2^>nul`) do (
        if not defined APP_PYTHON call :try_python "%%P" ""
    )
)
goto :eof

:try_python
rem %1 = 解释器路径，%2 = 额外参数；依赖自检通过才采用
if not exist "%~1" exit /b 1
if not defined PY_FOUND (
    set "PY_FOUND=1"
    set "PY_FIRST=%~1"
)
"%~1" %~2 -c "import app.main" >nul 2>nul
if errorlevel 1 exit /b 1
rem py 启动器会派生真正的解释器进程，这里解析为真实可执行文件，
rem 保证 PID 文件指向服务进程本身（否则 py.exe 退出后 stop 无法停止服务）
set "PY_REAL="
if "%~2"=="-3" (
    "%~1" -3 -c "import sys; print(sys.executable)" > "%TEMP%\indudoc-py-real.txt" 2>nul
    for /f "usebackq delims=" %%E in ("%TEMP%\indudoc-py-real.txt") do set "PY_REAL=%%E"
)
if defined PY_REAL (
    set "APP_PYTHON=%PY_REAL%"
    set "APP_PYTHON_ARGS="
    exit /b 0
)
set "APP_PYTHON=%~1"
set "APP_PYTHON_ARGS=%~2"
exit /b 0

:deps_missing
set "PYTHONIOENCODING=utf-8"
set "DEP_CHECK_LOG=%TEMP%\indudoc-dependency-check.log"
"%APP_PYTHON%" %APP_PYTHON_ARGS% -c "import app.main" >nul 2>"%DEP_CHECK_LOG%"
if not errorlevel 1 goto :eof
echo [错误] Python 依赖缺失或版本不兼容，无法启动服务。
echo 检测到的解释器：%APP_PYTHON% %APP_PYTHON_ARGS%
echo 请先安装依赖（脚本不会自动下载安装）：
echo   %APP_PYTHON% %APP_PYTHON_ARGS% -m pip install -r requirements.txt
echo 依赖自检输出：
"%PY_FIRST%" -c "import app.main" >nul 2>"%DEP_CHECK_LOG%"
type "%DEP_CHECK_LOG%"
exit /b 1

:check_not_running
rem START_BLOCKED: 0=可以启动 1=已在运行 2=端口被其它进程占用
set "START_BLOCKED=0"
call :read_pid
set "IS_ALIVE=0"
if defined TARGET_PID call :is_alive "%TARGET_PID%"
if defined TARGET_PID if "%IS_ALIVE%"=="1" (
    echo [InduDoc AI] 服务已在运行，未重复启动。
    echo   PID      : %TARGET_PID%
    echo   地址     : http://%HOST%:%PORT%
    echo   PID 文件 : %PID_FILE%
    set "START_BLOCKED=1"
    goto :eof
)
if defined TARGET_PID (
    echo [提示] 发现失效的 PID 文件（进程 %TARGET_PID% 不存在），已清理。
    del /q "%PID_FILE%" >nul 2>nul
)
call :detect_port_pid
if defined PORT_PID (
    echo [错误] 端口 %PORT% 已被进程 PID %PORT_PID% 占用，该进程不是由本脚本启动。
    echo 请先处理占用，或改用其他端口后重试：
    echo   1^) 查看占用进程：tasklist /FI "PID eq %PORT_PID%"
    echo   2^) 停止该进程   ：taskkill /PID %PORT_PID% /T /F
    echo   3^) 换端口启动   ：set PORT=8010 ^&^& start.bat start
    echo   4^) 若该服务就是本项目，请改用：start.bat status / start.bat stop
    set "START_BLOCKED=2"
)
goto :eof

:read_pid
set "TARGET_PID="
if not exist "%PID_FILE%" goto :eof
for /f "usebackq delims=" %%P in ("%PID_FILE%") do (
    if not defined TARGET_PID set "TARGET_PID=%%P"
)
goto :eof

:is_alive
set "IS_ALIVE=0"
tasklist /FI "PID eq %~1" /NH 2>nul | findstr /R /C:"[ ]%~1[ ]" >nul 2>nul
if not errorlevel 1 set "IS_ALIVE=1"
goto :eof

:detect_port_pid
set "PORT_PID="
for /f "tokens=1-5" %%A in ('netstat -ano -p tcp ^| findstr /R /C:":%PORT% " ^| findstr /C:"LISTENING"') do (
    if not defined PORT_PID set "PORT_PID=%%E"
)
if defined PORT_PID goto :eof
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "(Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue ^| Select-Object -First 1).OwningProcess" 2^>nul`) do (
    if not defined PORT_PID set "PORT_PID=%%P"
)
goto :eof
