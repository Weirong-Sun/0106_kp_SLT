#!/bin/bash
# 快速检查提取进度

echo "======================================================================"
echo "PHOENIX 关键点提取进度检查"
echo "======================================================================"
echo ""

# 检查运行进程
PID=2253891
if ps -p $PID > /dev/null 2>&1; then
    E_TIME=$(ps -o etime= -p $PID | tr -d ' ')
    echo "✓ 提取进程正在运行"
    echo "  进程 ID: $PID"
    echo "  运行时间: $E_TIME"
else
    echo "✗ 未找到运行中的提取进程"
fi

echo ""

# 检查文件
echo "文件状态:"
echo "----------------------------------------------------------------------"

for split in train dev test; do
    file="phoenix_keypoints.$split"
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        DATE=$(stat -c %y "$file" | cut -d' ' -f1,2 | cut -d'.' -f1)
        echo "  ✓ $file - $SIZE - $DATE"
    else
        echo "  ✗ $file - 不存在"
    fi
done

# 检查旧格式文件
if [ -f "phoenix_keypoints.pkl" ]; then
    SIZE=$(du -h phoenix_keypoints.pkl | cut -f1)
    DATE=$(stat -c %y phoenix_keypoints.pkl | cut -d' ' -f1,2 | cut -d'.' -f1)
    echo ""
    echo "旧格式文件:"
    echo "  ✓ phoenix_keypoints.pkl - $SIZE - $DATE"
fi

echo ""
echo "======================================================================"


