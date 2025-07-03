# Jay Tech Blog

[![Hugo](https://img.shields.io/badge/Hugo-0.147.9-blue.svg)](https://gohugo.io/)
[![Theme](https://img.shields.io/badge/Theme-PaperMod-green.svg)](https://github.com/adityatelange/hugo-PaperMod)
[![Deploy](https://github.com/Huijie-Liu/huijie-liu.github.io/actions/workflows/hugo.yaml/badge.svg)](https://github.com/Huijie-Liu/huijie-liu.github.io/actions/workflows/hugo.yaml)

> 专注于深度学习、计算机视觉和图形学的技术博客

## 🚀 快速开始

### 克隆项目

```bash
git clone https://github.com/Huijie-Liu/huijie-liu.github.io.git
cd huijie-liu.github.io

# 初始化主题子模块
git submodule update --init --recursive
```

### 本地开发

```bash
# 启动开发服务器
hugo server

# 包含草稿文章
hugo server -D

# 构建静态网站
hugo --gc
```

### 创建新文章

```bash
# 使用预设模板创建新文章
hugo new posts/my-new-post/index.md

# 编辑文章
# 文章位于 content/posts/my-new-post/index.md
```

## 📖 项目结构

```
huijie-liu.github.io/
├── archetypes/          # 内容模板
│   └── default.md      # 默认模板
├── content/            # 网站内容
│   └── posts/         # 博客文章
├── layouts/           # 自定义布局
│   └── partials/      # 部分模板
├── static/            # 静态资源
├── themes/            # 主题
│   └── PaperMod/      # PaperMod 主题 (git submodule)
└── hugo.yaml          # Hugo 配置文件
```

## 🛠️ 开发指南

### 文章编写

新文章会自动使用预设模板，包含：

- 完整的 Front Matter（标题、日期、标签等）
- 结构化的内容框架
- 代码示例区域
- 参考资料部分

### 图片资源

- 图片放在对应文章目录下的 `assets/` 文件夹
- 使用相对路径引用：`![描述](./assets/image.png)`

### 数学公式

支持 LaTeX 语法：

```markdown
行内公式：$E = mc^2$
块级公式：

$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

## 🚀 部署

### 自动部署

- 推送到 `main` 分支自动触发 GitHub Actions
- 自动构建并部署到 GitHub Pages

### 手动部署

```bash
# 构建网站
hugo --gc

# 部署到 GitHub Pages
# (通常由 GitHub Actions 自动处理)
```

## 🔧 配置说明

### 主要配置文件

- `hugo.yaml` - 网站主配置
- `layouts/partials/seo.html` - SEO 优化配置
- `layouts/partials/math.html` - 数学公式支持

### 环境要求

- Hugo v0.147.9+ (Extended 版本)
- Git (用于子模块管理)

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📝 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🔗 相关链接

- **网站**: <https://huijie-liu.github.io/>
- **Hugo 文档**: <https://gohugo.io/documentation/>
- **PaperMod 主题**: <https://github.com/adityatelange/hugo-PaperMod>

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues

---

**Jay Tech Blog** - 记录技术学习之路 🚀

