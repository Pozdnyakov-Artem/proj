@echo off
chcp 65001 >nul
echo Запуск контейнера с БД...
docker compose up -d db
if %errorlevel% neq 0 (
    echo Ошибка запуска Docker
    pause
    exit /b 1
)

echo Запуск приложения (Подождите)
python main.py

echo Остановка контейнера с БД...
docker compose down db
pause