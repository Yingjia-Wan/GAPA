#!/bin/bash

# 停止训练任务的脚本 - 停止所有相关进程和子进程

echo "🔍 正在查找运行中的训练任务及其子进程..."

# 查找所有相关进程（主进程和子进程）
# 包括: run_experiments.py, lora_training.py, lora_eval.py, 
#       prep_data_for_training.py, aggregate_run_seeds.py

MAIN_PIDS=$(pgrep -f "run_experiments.py")
TRAINING_PIDS=$(pgrep -f "lora_training.py")
EVAL_PIDS=$(pgrep -f "lora_eval.py")
PREP_PIDS=$(pgrep -f "prep_data_for_training.py")
AGGREGATE_PIDS=$(pgrep -f "aggregate_run_seeds.py")

# 合并所有进程ID（去重）
ALL_PIDS=$(echo "$MAIN_PIDS $TRAINING_PIDS $EVAL_PIDS $PREP_PIDS $AGGREGATE_PIDS" | tr ' ' '\n' | sort -u | tr '\n' ' ')

if [ -z "$ALL_PIDS" ]; then
    echo "❌ 没有找到运行中的训练任务"
    exit 0
fi

echo "📋 找到以下进程:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for pid in $ALL_PIDS; do
    if [ ! -z "$pid" ]; then
        ps -p $pid -o pid,cmd,etime 2>/dev/null | tail -n 1
    fi
done
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 查找所有子进程（通过进程树）
echo ""
echo "🌳 查找进程树中的子进程..."
CHILD_PIDS=""
for pid in $ALL_PIDS; do
    if [ ! -z "$pid" ]; then
        # 使用 pstree 或 pgrep 查找子进程
        children=$(pgrep -P $pid 2>/dev/null)
        if [ ! -z "$children" ]; then
            CHILD_PIDS="$CHILD_PIDS $children"
            echo "   进程 $pid 的子进程: $children"
        fi
    fi
done

# 合并所有进程（包括子进程）
if [ ! -z "$CHILD_PIDS" ]; then
    ALL_PIDS="$ALL_PIDS $CHILD_PIDS"
    ALL_PIDS=$(echo "$ALL_PIDS" | tr ' ' '\n' | sort -u | tr '\n' ' ')
fi

TOTAL_COUNT=$(echo $ALL_PIDS | wc -w)
echo ""
echo "📊 总共找到 $TOTAL_COUNT 个进程需要停止"

echo ""
read -p "⚠️  是否要停止所有这些进程? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🛑 正在停止进程..."
    
    # 首先尝试优雅停止（SIGTERM）
    for pid in $ALL_PIDS; do
        if [ ! -z "$pid" ] && kill -0 $pid 2>/dev/null; then
            echo "   发送 SIGTERM 到进程 $pid..."
            kill $pid 2>/dev/null
        fi
    done
    
    # 等待进程退出
    echo "   等待进程退出..."
    sleep 3
    
    # 检查是否还有进程在运行
    REMAINING=""
    for pid in $ALL_PIDS; do
        if [ ! -z "$pid" ] && kill -0 $pid 2>/dev/null; then
            REMAINING="$REMAINING $pid"
        fi
    done
    
    # 如果还有进程在运行，强制停止
    if [ ! -z "$REMAINING" ]; then
        echo "   ⚠️  部分进程仍在运行，强制停止..."
        for pid in $REMAINING; do
            if [ ! -z "$pid" ] && kill -0 $pid 2>/dev/null; then
                echo "   发送 SIGKILL 到进程 $pid..."
                kill -9 $pid 2>/dev/null
            fi
        done
        sleep 1
    fi
    
    # 最终检查
    FINAL_REMAINING=$(pgrep -f "run_experiments.py|lora_training.py|lora_eval.py|prep_data_for_training.py|aggregate_run_seeds.py")
    if [ -z "$FINAL_REMAINING" ]; then
        echo ""
        echo "✅ 所有训练任务已成功停止"
    else
        echo ""
        echo "⚠️  警告: 以下进程可能仍在运行:"
        echo "$FINAL_REMAINING"
        echo "   可以手动运行: kill -9 $FINAL_REMAINING"
    fi
else
    echo "❌ 取消操作"
fi

