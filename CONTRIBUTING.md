<!-- DANSWER_METADATA={"link": "https://github.com/onyx-dot-app/onyx/blob/main/CONTRIBUTING.md"} -->

# 为 Onyx 做贡献

嘿！我们很高兴你对 Onyx 感兴趣。

作为一个快速发展领域中的开源项目，我们欢迎所有形式的贡献。

## 💃 指南

### 贡献机会

[GitHub Issues](https://github.com/onyx-dot-app/onyx/issues) 页面是寻找贡献想法的好地方。

已经被维护者明确批准的 issues（与项目方向一致的）将会标记为 `approved by maintainers` 标签。
标记为 `good first issue` 的 issues 是特别适合新手入门的。

**连接器**是另一个很好的贡献切入点。详情请参考这个
[README.md](https://github.com/onyx-dot-app/onyx/blob/main/backend/onyx/connectors/README.md)。

如果你有新的/不同的贡献想法，我们很乐意听取！
你的意见对确保 Onyx 朝着正确的方向发展至关重要。
在开始实施之前，请先提出一个 GitHub issue。

随时欢迎通过 [Slack](https://join.slack.com/t/danswer/shared_invite/zt-1w76msxmd-HJHLe3KNFIAIzk_0dSOKaQ) 或
[Discord](https://discord.gg/TDJ59cGV2X) 直接联系我们（Chris Weaver / Yuhong Sun）讨论任何问题。

### 贡献代码

要为此项目贡献代码，请遵循["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)工作流程。
在开启 pull request 时，请提及相关的 issues 并随时标记相关的维护者。

在创建 pull request 之前，请确保新的更改符合格式化和代码检查要求。
有关如何在本地运行这些检查的说明，请参见[格式化和代码检查](#formatting-and-linting)部分。

### 获取帮助 🙋

我们的目标是让贡献变得尽可能容易。如果你遇到任何问题，请不要犹豫，尽管联系我们。
这样我们就可以帮助未来的贡献者和用户避免同样的问题。

我们在 [Slack](https://join.slack.com/t/danswer/shared_invite/zt-1w76msxmd-HJHLe3KNFIAIzk_0dSOKaQ)
和 [Discord](https://discord.gg/TDJ59cGV2X) 上有支持频道和一般性的有趣讨论。

我们期待在那里见到你！

## 开始使用 🚀

作为一个完整功能的应用，Onyx 依赖于一些外部软件，具体包括：

- [Postgres](https://www.postgresql.org/)（关系型数据库）
- [Vespa](https://vespa.ai/)（向量数据库/搜索引擎）
- [Redis](https://redis.io/)（缓存）
- [Nginx](https://nginx.org/)（通常开发流程中不需要）

> **注意：**
> 本指南提供了使用 Docker 容器来提供上述外部软件的本地源码构建和运行 Onyx 的说明。我们认为这种组合更适合
> 开发目的。如果你更喜欢使用预构建的容器镜像，我们在下面提供了在 Docker 中运行完整 Onyx 堆栈的说明。

### 本地设置

请确保使用 Python 3.11 版本。有关在 macOS 上安装 Python 3.11 的说明，请参阅 [CONTRIBUTING_MACOS.md](./CONTRIBUTING_MACOS.md) 自述文件。

如果使用较低版本，则需要对代码进行修改。
如果使用较高版本，有时某些库将不可用（例如，我们过去在使用较高版本的 Python 时遇到了 Tensorflow 的问题）。

#### 后端：Python 需求

目前，我们使用 pip 并推荐创建一个虚拟环境。

为了方便，这里有一个命令：

```bash
python -m venv .venv
source .venv/bin/activate
```

> **注意：**
> 如果你计划在某些 IDE 中使用 mypy，则此虚拟环境不能在 onyx 目录内设置。
> 为了简单起见，我们建议在 onyx 目录外设置虚拟环境。

_对于 Windows，请使用命令提示符激活虚拟环境：_

```bash
.venv\Scripts\activate
```

如果使用 PowerShell，命令略有不同：

```powershell
.venv\Scripts\Activate.ps1
```

安装所需的 Python 依赖项：

```bash
pip install -r onyx/backend/requirements/default.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install -r onyx/backend/requirements/dev.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install -r onyx/backend/requirements/ee.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install -r onyx/backend/requirements/model_server.txt -i https://mirrors.aliyun.com/pypi/simple/
```

为 Python 安装 Playwright（Web 连接器所需的无头浏览器）

在激活的 Python 虚拟环境中，通过运行以下命令安装 Playwright for Python：

```bash
playwright install
```

你可能需要停用并重新激活你的虚拟环境，以便 `playwright` 出现在你的��径中。

#### 前端：Node 依赖项

为前端安装 [Node.js 和 npm](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)。
完成上述操作后，导航到 `onyx/web` 运行：

```bash
npm i
```

#### 外部软件的 Docker 容器

你需要安装 Docker 才能运行这些容器。

首先导航到 `onyx/deployment/docker_compose`，然后启动 Postgres/Vespa/Redis：

```bash
docker compose -f docker-compose.dev.yml -p onyx-stack up -d index relational_db cache
```

（index 指的是 Vespa，relational_db 指的是 Postgres，cache 指的是 Redis）

#### 本地运行 Onyx

要启动前端，请导航到 `onyx/web` 并运行：

```bash
npm run dev
```

接下来，启动运行本地 NLP 模型的模型服务器。
导航到 `onyx/backend` 并运行：

```bash
uvicorn model_server.main:app --reload --port 9000
```

_对于 Windows（兼容 PowerShell 和命令提示符）：_

```bash
powershell -Command "uvicorn model_server.main:app --reload --port 9000"
```

第一次运行 Onyx 时，你需要为 Postgres 运行数据库迁移。
在第一次之后，除非数据库模型发生变化，否则不再需要这样做。

导航到 `onyx/backend` 并在虚拟环境激活的情况下运行：

```bash
alembic upgrade head
```

接下来，启动协调后台作业的任务队列。
需要更多时间的作业将异步运行，不在 API 服务器中运行。

仍然在 `onyx/backend` 目录中，运行：

```bash
python ./scripts/dev_run_background_jobs.py
```

要运行后端 API 服务器，请返回到 `onyx/backend` 并运行：

```bash
AUTH_TYPE=disabled uvicorn onyx.main:app --reload --port 8080
```

_对于 Windows（兼容 PowerShell 和命令提示符）：_

```bash
powershell -Command "
    $env:AUTH_TYPE='disabled'
    uvicorn onyx.main:app --reload --port 8080
"
```

> **注意：**
> 如果你需要更详细的日志记录，请为相关服务添加额外的环境变量 `LOG_LEVEL=DEBUG`。

#### 收尾工作

你现在应该有 4 个服务器在运行：

- Web 服务器
- 后端 API
- 模型服务器
- 后台作业

现在，在浏览器中访问 `http://localhost:3000`。你应该会看到 Onyx 入门向导，在这里你可以将你的外部 LLM 提供商连接到 Onyx。

你已经成功设置了一个本地的 Onyx 实例！ 🏁

#### 在容器中运行 Onyx 应用程序

你可以从预构建的镜像运行完整的 Onyx 应用程序堆栈，包括所有外部软件依赖项。

导航到 `onyx/deployment/docker_compose` 并运行：

```bash
docker compose -f docker-compose.dev.yml -p onyx-stack up -d
```

在 Docker 拉取并启动这些容器后，导航到 `http://localhost:3000` 使用 Onyx。

如果你想对 Onyx 进行更改并在 Docker 中运行这些更改，你还可以构建一个包含你更改的本地版本的 Onyx 容器镜像，如下所示：

```bash
docker compose -f docker-compose.dev.yml -p onyx-stack up -d --build
```

### 格式化和代码检查

#### 后端

对于后端，你需要设置 pre-commit hooks（black / reorder-python-imports）。
首先，按照[此处](https://pre-commit.com/#installation)的说明安装 pre-commit（如果你还没有安装）。

在虚拟环境激活的情况下，安装 pre-commit 库：

```bash
pip install pre-commit
```

然后，从 `onyx/backend` 目录运行：

```bash
pre-commit install
```

此外，我们使用 `mypy` 进行静态类型检查。
Onyx 是完全类型注释的，我们希望保持这种状态！
要手动运行 mypy 检查，请从 `onyx/backend` 目录运行 `python -m mypy .`。

#### Web

我们使用 `prettier` 进行格式化。所需版本（2.8.8）将通过 `onyx/web` 目录中的 `npm i` 安装。
要运行格式化程序，请从 `onyx/web` 目录使用 `npx prettier --write .`。
在创建 pull request 之前，请仔细检查 prettier 是否通过。

### 发布流程

Onyx 大致遵循 SemVer 版本控制标准。
重大更改会发布一个“次要”版本更新。目前我们使用补丁版本来表示小的功能更改。
每个标签都会自动推送一组 Docker 容器到 DockerHub。
你可以在[这里](https://hub.docker.com/search?q=onyx%2F)查看这些容器。
