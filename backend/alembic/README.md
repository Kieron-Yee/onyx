<!-- DANSWER_METADATA={"link": "https://github.com/onyx-dot-app/onyx/blob/main/backend/alembic/README.md"} -->

# Alembic 数据库迁移

这些文件用于在关系数据库（Postgres）中创建/更新表。
Onyx 迁移使用通用的单数据库配置和异步 dbapi。

## 生成新的迁移：

在 onyx/backend 目录下运行：
`alembic revision --autogenerate -m <迁移描述>`

更多信息可以在这里找到：https://alembic.sqlalchemy.org/en/latest/autogenerate.html

## 运行迁移

运行所有未应用的迁移：
`alembic upgrade head`

撤销迁移：
`alembic downgrade -X`
其中 X 是您要从当前状态撤销的迁移数量
