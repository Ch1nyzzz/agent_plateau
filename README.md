# agent_plateau

## 在线周报网站
- 网站地址：https://ch1nyzzz.github.io/agent_plateau/ ，可直接在线查看最新周报。
- 每次推送 `docs/` 下的 Markdown 文件，GitHub Actions 会自动部署更新网站。
- 若需本地预览，可先运行 `pip install mkdocs-material python-markdown-math`，再执行 `mkdocs serve`。

## 更新周报流程
1. 在 `docs/` 中新增如 `week4.md` 的周报文件，并填写内容。
2. 如需在导航栏展示，更新 `mkdocs.yml` 里的 `nav` 配置。
3. 执行 `git add . && git commit -m "Add week 4 report" && git push`，等待 GitHub Actions 完成部署。
