#!/bin/bash
# Cursor Sync Changes SSH 认证配置脚本
# 用于解决 Cursor 中 git pull/push 时的 SSH 认证问题

echo "========================================="
echo "配置 SSH 密钥以支持 Cursor Sync Changes"
echo "========================================="
echo ""

# 1. 启动 SSH agent（如果未运行）
if [ -z "$SSH_AUTH_SOCK" ]; then
    echo "启动 SSH agent..."
    eval "$(ssh-agent -s)"
else
    echo "SSH agent 已在运行: $SSH_AUTH_SOCK"
fi

# 2. 添加私钥到 agent
echo ""
echo "添加 SSH 密钥到 agent..."
echo "注意：系统会提示你输入私钥密码（passphrase）"
ssh-add ~/.ssh/id_ed25519

# 3. 检查密钥是否已添加
echo ""
echo "检查已加载的密钥："
ssh-add -l

# 4. 测试 GitHub 连接
echo ""
echo "测试 GitHub 连接..."
ssh -T git@github.com 2>&1 | head -2

echo ""
echo "========================================="
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "✅ SSH 配置成功！现在可以在 Cursor 中使用 Sync Changes 了"
else
    echo "❌ SSH 连接失败，请检查："
    echo "   1. 私钥密码是否正确"
    echo "   2. 公钥是否已添加到 GitHub (https://github.com/settings/keys)"
    echo "   3. 你的公钥: $(cat ~/.ssh/id_ed25519.pub | cut -d' ' -f1-2)"
fi
echo "========================================="



