#!/bin/bash
# 切换到 HTTPS 方式（推荐用于 Cursor Sync Changes）

echo "========================================="
echo "将 Git 远程仓库切换到 HTTPS 方式"
echo "========================================="
echo ""

# 当前远程 URL
echo "当前远程仓库 URL："
git remote -v
echo ""

# 切换到 HTTPS
echo "切换到 HTTPS..."
git remote set-url origin https://github.com/Weirong-Sun/0106_kp_SLT.git

echo ""
echo "更新后的远程仓库 URL："
git remote -v

echo ""
echo "========================================="
echo "✅ 已切换到 HTTPS 方式"
echo ""
echo "下次 git pull/push 时需要："
echo "  - 用户名：Weirong-Sun"
echo "  - 密码：个人访问令牌（Personal Access Token）"
echo ""
echo "创建个人访问令牌："
echo "  https://github.com/settings/tokens"
echo "  权限：勾选 'repo'（完整仓库权限）"
echo ""
echo "配置凭据存储（可选，避免每次输入）："
echo "  git config --global credential.helper store"
echo "========================================="

