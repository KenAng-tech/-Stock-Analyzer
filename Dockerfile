FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# 创建应用用户
RUN useradd -m stock_analyzer
RUN mkdir -p /app/data /app/logs
RUN chown -R stock_analyzer:stock_analyzer /app

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 切换到应用用户
USER stock_analyzer

# 暴露端口
EXPOSE 5002

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5002/api/health || exit 1

# 启动命令
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]