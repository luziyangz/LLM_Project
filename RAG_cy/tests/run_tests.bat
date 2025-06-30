@echo off
chcp 65001
echo PDF解析器测试套件
echo ========================

cd /d "%~dp0"

echo 检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: Python未安装或不在PATH中
    pause
    exit /b 1
)

echo.
echo 检查依赖库...
python -c "import docling; print('✓ Docling库已安装')"
if errorlevel 1 (
    echo 错误: Docling库未安装
    echo 请运行: pip install docling
    pause
    exit /b 1
)

echo.
echo 开始运行测试...
python test_pdf_parsing.py

echo.
echo 测试完成，按任意键退出...
pause