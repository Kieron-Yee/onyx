<!-- DANSWER_METADATA={"link": "https://github.com/onyx-dot-app/onyx/blob/main/README.md"} -->

<a name="readme-top"></a>

<h2 align="center">
<a href="https://www.onyx.app/"> <img width="50%" src="https://github.com/onyx-dot-app/onyx/blob/logo/OnyxLogoCropped.jpg?raw=true)" /></a>
</h2>

<p align="center">
<p align="center">开源生成式AI + 企业搜索.</p>

<p align="center">
<a href="https://docs.onyx.app/" target="_blank">
    <img src="https://img.shields.io/badge/docs-view-blue" alt="Documentation">
</a>
<a href="https://join.slack.com/t/onyx-dot-app/shared_invite/zt-2twesxdr6-5iQitKZQpgq~hYIZ~dv3KA" target="_blank">
    <img src="https://img.shields.io/badge/slack-join-blue.svg?logo=slack" alt="Slack">
</a>
<a href="https://discord.gg/TDJ59cGV2X" target="_blank">
    <img src="https://img.shields.io/badge/discord-join-blue.svg?logo=discord&logoColor=white" alt="Discord">
</a>
<a href="https://github.com/onyx-dot-app/onyx/blob/main/README.md" target="_blank">
    <img src="https://img.shields.io/static/v1?label=license&message=MIT&color=blue" alt="License">
</a>
</p>

<strong>[Onyx](https://www.onyx.app/)</strong> (前身为Danswer)是一个连接到您公司文档、应用程序和人员的AI助手。
Onyx提供聊天界面，并可以连接任何您选择的LLM模型。Onyx可以部署在任何地方���适应任何规模
- 从笔记本电脑到本地服务器或云端。由于您拥有部署的所有权，您的用户数据和聊天记录完全在您的
控制之下。Onyx采用双重许可，大部分内容遵循MIT许可证，设计模块化且易于扩展。该系统还完全
适用于生产环境，具备用户认证、角色管理(管理员/基本用户)、聊天持久化，以及用于
配置AI助手的用户界面。

Onyx还可以作为跨所有常用工作场所工具(如Slack、Google Drive、Confluence等)的企业搜索工具。
通过结合LLM和团队特定知识，Onyx成为团队的专业知识专家。想象一下如果ChatGPT
能够访问您团队的独特知识会是什么样！它可以回答诸如"客户想要功能X，这是否已经
支持？"或"功能Y的拉取请求在哪里？"等问题。

<h3>使用方式</h3>

Onyx网页应用：

https://github.com/onyx-dot-app/onyx/assets/32520769/563be14c-9304-47b5-bf0a-9049c2b6f410

或者，将Onyx集成到您现有的Slack工作流程中（更多集成即将推出😁）：

https://github.com/onyx-dot-app/onyx/assets/25087905/3e19739b-d178-4371-9a38-011430bdec1b

关于管理连接器和用户的管理界面的更多详情，请查看我们的
<strong><a href="https://www.youtube.com/watch?v=geNzY1nbCnU">完整视频演示</a></strong>！

## 部署

Onyx可以轻松地在本地运行（甚至在笔记本电脑上）或通过单个
`docker compose`命令部署在虚拟机上。查看我们的[文档](https://docs.onyx.app/quickstart)了解更多信息。

我们还内置了对Kubernetes部署的支持。相关文件可以在[这里](https://github.com/onyx-dot-app/onyx/tree/main/deployment/kubernetes)找到。

## 💃 主要特性

- 具备选择文档进行对话功能的聊天界面
- 创建具有不同提示和知识库支持的自定义AI助手
- 连接您选择的LLM（自托管可实现完全离线解决方案）
- 文档搜索 + 自然语言查询的AI回答
- 连接所有常用工作场所工具，如Google Drive、Confluence、Slack等
- Slack集成，直接在Slack中获取答案和搜索结果

## 🚧 路线图

- 与特定团队成员和用户组共享聊天/提示
- 多模态模型支持，可与图像、视频等进行对话
- 在聊天会话中选择LLM和参数
- 工具调用和代理配置选项
- 组织理解能力，可以定位和推荐团队中的专家

## Onyx的其他显著优势

- 具有文档级访问管理的用户认证
- 跨所有来源的最佳混合搜索（BM-25 + 前缀感知嵌入模型）
- 用于配置连接器、文档集、访问权限等的管理仪表板
- 自定义深度学习模型 + 从用户反馈中学习
- 便捷部署，可以在任何您选择的地方托管Onyx

## 🔌 连接器

高效获取最新更改，支持：

- Slack
- GitHub
- Google Drive
- Confluence
- Jira
- Zendesk
- Gmail
- Notion
- Gong
- Slab
- Linear
- Productboard
- Guru
- Bookstack
- Document360
- Sharepoint
- Hubspot
- Local Files
- Websites
- And more ...

## 📚 版本

Onyx提供两个版本：

- Onyx社区版 (CE) 在MIT Expat许可证下免费提供。该版本包含上述所有核心功能。如果您按照上述部署指南操作，这是您将获得的Onyx版本。
- Onyx企业版 (EE) 包含主要适用于大型组织的额外功能。具体包括：
  - 单点登录 (SSO)，支持SAML和OIDC
  - 基于角色的访问控制
  - 从连接源继承的文档权限
  - 管理员可访问的使用分析和查询历史
  - 白标定制
  - API密钥认证
  - 密钥加密
  - 以及更多！查看[我们的网站](https://www.onyx.app/)了解最新信息。

试用Onyx企业版：

1. 查看我们的[云产品](https://cloud.onyx.app/signup)。
2. 如需自托管，请通过[founders@onyx.app](mailto:founders@onyx.app)联系我们或在[Cal](https://cal.com/team/danswer/founders)上预约与我们通话。

## 💡 贡献

想要贡献代码？请查看[贡献指南](CONTRIBUTING.md)了解更多详情。

## ⭐Star历史

[![Star History Chart](https://api.star-history.com/svg?repos=onyx-dot-app/onyx&type=Date)](https://star-history.com/#onyx-dot-app/onyx&Date)
