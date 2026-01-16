# Git SSH 认证问题解决方案

## 问题描述
在 Cursor 中执行 `git push` 时遇到错误：
```
Permission denied (publickey)
```

## 问题原因
1. **私钥有密码保护**：`id_ed25519` 私钥设置了 passphrase（密码），SSH agent 无法自动加载
2. **SSH agent 未加载密钥**：当前 shell 环境中的 SSH agent 可能没有加载私钥

## 解决方案

### 方案1：手动添加密钥到 SSH agent（推荐）

在终端中执行以下命令：

```bash
# 1. 启动 SSH agent（如果未运行）
eval "$(ssh-agent -s)"

# 2. 添加私钥到 agent（会提示输入密码）
ssh-add ~/.ssh/id_ed25519

# 3. 测试 GitHub 连接
ssh -T git@github.com
```

如果连接成功，你会看到：
```
Hi Weirong-Sun! You've successfully authenticated, but GitHub does not provide shell access.
```

### 方案2：确保公钥已添加到 GitHub

你的公钥内容：
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAII2CojAwfiopbJYNjGVIfeycd20o9tIug1hgghpSXUfy swr0311@outlook.com
```

**检查步骤：**
1. 登录 GitHub：https://github.com/settings/keys
2. 查看 "SSH keys" 列表
3. 确认上述公钥是否存在

**如果公钥不存在，添加步骤：**
1. 点击 "New SSH key"
2. 标题：填写一个描述（如 "My Server"）
3. 密钥类型：选择 "Authentication Key"
4. 密钥内容：复制上面的公钥内容
5. 点击 "Add SSH key"

### 方案3：使用 SSH config 自动加载密钥（永久解决）

如果你希望每次启动时自动加载密钥，可以创建一个启动脚本：

```bash
# 添加到 ~/.bashrc 或 ~/.profile
if [ -z "$SSH_AUTH_SOCK" ]; then
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
fi
```

**注意：** 这会在每次打开终端时提示输入密码。

### 方案4：使用无密码的密钥对（不推荐）

如果你不希望每次都要输入密码，可以：
1. 生成一个无密码的新密钥对
2. 将新公钥添加到 GitHub
3. 在 SSH config 中指定使用新密钥

```bash
# 生成无密码密钥（不设置 passphrase）
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_nopass -C "swr0311@outlook.com" -N ""

# 添加新公钥到 GitHub
cat ~/.ssh/id_ed25519_nopass.pub
```

然后在 `~/.ssh/config` 中更新：
```
Host github.com
  HostName github.com
  User git
  IdentityFile /root/.ssh/id_ed25519_nopass
  IdentitiesOnly yes
```

### 方案5：临时使用 HTTPS 方式（快速解决）

如果急需推送代码，可以临时切换到 HTTPS：

```bash
# 更改远程仓库 URL
git remote set-url origin https://github.com/Weirong-Sun/0106_kp_SLT.git

# 推送（会提示输入用户名和密码或个人访问令牌）
git push
```

**注意：** HTTPS 方式需要：
- GitHub 用户名
- 个人访问令牌（Personal Access Token），而不是密码

## 验证步骤

执行以下命令验证是否解决：

```bash
# 1. 检查 SSH agent 中的密钥
ssh-add -l

# 2. 测试 GitHub 连接
ssh -T git@github.com

# 3. 尝试推送
git push
```

## 常见问题

### Q1: 为什么私钥有密码？
- 提高安全性，即使私钥泄露也需要密码才能使用

### Q2: 可以在 Cursor 中自动输入密码吗？
- 不能，密码输入需要在交互式终端中进行

### Q3: 如何避免每次都要输入密码？
- 方案3：将密钥添加到 SSH agent 后，在同一个会话中不需要重复输入
- 方案4：使用无密码密钥（安全性较低）

## 推荐做法

1. **当前会话**：使用方案1，在终端中手动添加密钥
2. **长期使用**：确保公钥已添加到 GitHub（方案2）
3. **自动化**：如果需要，使用方案3配置自动加载（仍需输入密码）

## 当前状态

- ✅ SSH config 配置正确
- ✅ 私钥权限正确（600）
- ✅ 公钥文件存在
- ❌ SSH agent 未加载私钥（需要输入密码）
- ❓ 需要确认公钥是否已添加到 GitHub

## Cursor Sync Changes 特殊说明

**问题：** 在 Cursor 的 "Sync Changes" 操作中，Git 会执行 `git pull --tags origin master`，但由于私钥有密码，无法自动加载密钥，导致认证失败。

**解决方案选择：**

### 方案A：使用终端手动添加密钥（临时解决）

在 Cursor 的终端中运行配置脚本：

```bash
# 运行配置脚本
./setup_ssh_for_cursor.sh
```

或手动执行：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519  # 会提示输入密码
ssh -T git@github.com      # 测试连接
```

**注意：** 此方法只在当前终端会话有效，关闭终端后需要重新添加。

### 方案B：切换到 HTTPS（推荐用于 Cursor）

对于 Cursor 的自动同步操作，使用 HTTPS 更可靠（不需要交互输入密码）：

```bash
# 1. 更改远程仓库 URL 为 HTTPS
git remote set-url origin https://github.com/Weirong-Sun/0106_kp_SLT.git

# 2. 配置 Git 凭据存储（可选，避免每次都输入）
git config --global credential.helper store
```

**首次推送时需要：**
- GitHub 用户名：`Weirong-Sun`
- 个人访问令牌（Personal Access Token）：不是密码！

**创建个人访问令牌：**
1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 勾选权限：`repo`（完整仓库权限）
4. 生成后复制令牌（只显示一次）
5. 在 Git 提示时，用户名输入 GitHub 用户名，密码输入令牌

### 方案C：使用无密码密钥（长期解决）

如果希望 Cursor 可以自动同步，可以创建一个专门用于 Cursor 的无密码密钥：

```bash
# 1. 生成无密码密钥
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_cursor -C "cursor@github" -N ""

# 2. 查看公钥并添加到 GitHub
cat ~/.ssh/id_ed25519_cursor.pub

# 3. 更新 SSH config 指定使用新密钥
cat >> ~/.ssh/config << EOF

Host github.com-cursor
  HostName github.com
  User git
  IdentityFile /root/.ssh/id_ed25519_cursor
  IdentitiesOnly yes
EOF

# 4. 更新 Git 远程 URL
git remote set-url origin git@github.com-cursor:Weirong-Sun/0106_kp_SLT.git
```

**注意：** 此方法安全性较低（密钥无密码），但适合自动化场景。

## 推荐方案（针对 Cursor）

1. **如果只是偶尔需要同步**：使用方案A，在终端手动添加密钥
2. **如果经常使用 Cursor 同步**：使用方案B（HTTPS + 个人访问令牌），最可靠
3. **如果需要完全自动化**：使用方案C（无密码密钥），但需注意安全

