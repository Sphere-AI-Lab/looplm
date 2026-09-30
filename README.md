# LoopLM

Project page for **What Makes Recurrence Effective in Looped Language Models?**

Xinlin Zhuang, Siyuan Wang, Imran Razzak, and Weiyang Liu · CUHK / MBZUAI

目标地址：**https://spherelab.ai/looplm/**

页面参考 [SphereLab Pion 项目页](https://spherelab.ai/pion/) 的学术排版，以论文源码 `iclr2027_looplm_analysis/arxiv.tex` 和 `secs/` 为内容依据。使用原始实验图表，不依赖前端框架、构建步骤、CDN 或在线字体服务。

## 本地预览

```bash
cd /Users/xinlin.zhuang/Codes/looplm
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

浏览器打开 **http://localhost:8000/**，终端按 `Ctrl+C` 停止服务。也可以直接打开 `docs/index.html`；剪贴板功能在本地文件环境受浏览器限制时会自动改为选中文本并提示手动复制。

## 发布到 spherelab.ai/looplm/

已于 2026-09-29 检查组织主页的公开 [CNAME 文件](https://github.com/Sphere-AI-Lab/Sphere-AI-Lab.github.io/blob/main/CNAME)，内容为 `spherelab.ai`；该域名的 HTTP 响应也显示由 GitHub Pages 托管。

GitHub Pages 的规则是：组织主页绑定自定义域名后，同一组织下未单独设置域名的项目站点会继承该域名。例如，`Sphere-AI-Lab/looplm` 的 Pages 对应 `https://spherelab.ai/looplm/`。见 [GitHub 官方说明：跨仓库使用自定义域名](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site/about-custom-domains-and-github-pages#using-a-custom-domain-across-multiple-repositories)。

### 1. 提交并推送网站文件

在本地确认页面内容后执行（当前实现没有自动提交或推送）：

```bash
cd /Users/xinlin.zhuang/Codes/looplm
git add README.md .gitignore docs scripts
git commit -m "Add LoopLM research project page"
git push origin main
```

### 2. 开启仓库的 GitHub Pages

由有仓库设置权限的人打开：

**https://github.com/Sphere-AI-Lab/looplm/settings/pages**

在 **Build and deployment** 中设置：

- **Source**：`Deploy from a branch`
- **Branch**：`main`
- **Folder**：`/docs`
- 点击 **Save**。

此方案直接发布静态文件，无需另外设置 GitHub Actions 工作流。`docs/.nojekyll` 已禁用 Jekyll 处理。[官方配置说明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

### 3. 等待并检查上线

在仓库 **Actions** 中查看 Pages 部署任务；成功后访问 **https://spherelab.ai/looplm/**。首次发布可能需要几分钟，GitHub 提示最长可到 10 分钟。后续每次推送 `main` 分支中的 `docs/` 修改都会触发重新部署。

**本仓库不要新建 `CNAME`，也不要在 Custom domain 中填写 `spherelab.ai/looplm`。** `/looplm/` 是 URL 路径，不是域名，直接由仓库名决定。组织主页已经管理 `spherelab.ai`，本项目无需修改 DNS。Custom domain 字段若显示继承的组织域名，不要覆盖它。

如果显示 404，按顺序检查：Pages 是否选择 `main` + `/docs`、最新部署是否成功、`docs/index.html` 是否已推送、仓库是否属于 `Sphere-AI-Lab`，以及组织主页是否仍绑定 `spherelab.ai`。私有仓库还需满足对应 GitHub 套餐的 Pages 支持条件。

## 更新内容

| 文件 | 用途 |
| --- | --- |
| `docs/index.html` | 标题、作者、摘要、实验结论、按钮链接和 BibTeX |
| `docs/styles.css` | 酒红色主题、字体、桌面/移动端布局 |
| `docs/site.js` | 循环次数演示、图片放大、导航高亮、复制引用 |
| `docs/assets/figures/` | 网页 PNG 和可下载的原始 PDF 图表 |
| `docs/assets/figures/manifest.json` | 图表源文件和导出尺寸 |
| `scripts/export_figures.py` | 从论文素材重新导出图表 |

Paper 已链接到 [arXiv:2609.36636](https://arxiv.org/abs/2609.36636)，页面底部 BibTeX 同步包含 arXiv 编号和论文 URL。

Code 仍显示不可点击的 `coming soon`。本仓库仅用于项目页面；代码发布后，再将 Code 状态替换为实际代码仓库链接。

### 重新导出图表（仅开发需要）

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pymupdf
python3 scripts/export_figures.py ../iclr2027_looplm_analysis
```

这会复制七组原始 PDF，并导出 2400 像素宽的 PNG。网页本身没有 Python 或 PyMuPDF 运行依赖。若导出后的图片比例变化，同步更新 `index.html` 中对应图片的 `width` / `height`，避免加载时页面跳动。网站资源使用相对路径，可直接部署在 `/looplm/` 子路径。

说明：源论文 `secs/motivation.tex` 的个别配置标签与图注存在不一致；网页按 `secs/preliminaries.tex` 的图注将 28.52 → 31.49 的结果标为 BaseLoop 4 × 5，并统一采用“物理层数 × 循环次数”的展示方式。请在论文定稿时一并核对。
