// PyTorch Daily - build static site from markdown files
// 架构：目录页 + 每篇文章独立页面（按需加载，文章再多首屏也轻量）
// Usage: node build_site.js
const fs = require("fs");
const path = require("path");
const { marked } = require("marked");
const hljs = require("highlight.js");

const SRC_DIR = __dirname;
const OUT_DIR = path.join(__dirname, "site");
const POSTS_DIR = path.join(OUT_DIR, "posts");

// 收集 markdown 文件（排除构建脚本与 site 目录），按日期倒序
const files = fs
  .readdirSync(SRC_DIR)
  .filter((f) => f.endsWith(".md"))
  .sort()
  .reverse(); // 最新在前（文件名以日期开头）

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ---------- 代码高亮（构建期完成，页面无需加载任何 JS） ----------
// 必须在 marked.parse 之前注册
// marked v18 对象式 renderer：code token 接收 { text, lang }
const renderer = {
  code({ text, lang }) {
    const language = (lang || "").trim().split(/\s+/)[0];
    let highlighted;
    if (language && hljs.getLanguage(language)) {
      highlighted = hljs.highlight(text, {
        language,
        ignoreIllegals: true,
      }).value; // hljs 输出已做 HTML 转义
    } else {
      // 无语言标记：先尝试自动检测，失败则纯文本输出
      const auto = hljs.highlightAuto(text);
      highlighted = auto.relevance > 3 ? auto.value : esc(text);
    }
    const cls = language ? ` class="hljs language-${language}"` : ' class="hljs"';
    return `<pre><code${cls}>${highlighted}</code></pre>`;
  },
};
marked.use({ renderer });

function extractMeta(html, filename) {
  const date = filename.slice(0, 10);
  const h1 = (html.match(/<h1[^>]*>(.*?)<\/h1>/) || [])[1] || filename;
  const h2 = (html.match(/<h2[^>]*>(.*?)<\/h2>/) || [])[1] || "";
  const title = h1.replace(/<[^>]+>/g, "").replace(/PyTorch 每日一课 · /, "");
  const subtitle = h2.replace(/<[^>]+>/g, "");
  return { date, title, subtitle };
}

const articles = files.map((f) => {
  const md = fs.readFileSync(path.join(SRC_DIR, f), "utf-8");
  const html = marked.parse(md, { gfm: true, breaks: false });
  const meta = extractMeta(html, f);
  // slug 直接复用文件名（YYYY-MM-DD_topic 形式，URL 安全）
  const slug = f.replace(/\.md$/, "").replace(/[^a-z0-9_-]/gi, "");
  return { file: f, html, meta, slug, page: `posts/${slug}.html` };
});

// ---------- 共享样式 ----------
const CSS = `
  :root {
    --bg: #f6f7f9;
    --card: #ffffff;
    --text: #1a1d21;
    --muted: #6b7280;
    --accent: #ee4d2d;
    --border: #e5e7eb;
    --code-bg: #f3f4f6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #111318;
      --card: #1a1d23;
      --text: #e6e8eb;
      --muted: #9ca3af;
      --border: #2a2e36;
      --code-bg: #232730;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Segoe UI", sans-serif;
    line-height: 1.75;
    -webkit-text-size-adjust: 100%;
  }
  .wrap { max-width: 720px; margin: 0 auto; padding: 16px 16px 60px; }
  header.site {
    background: linear-gradient(135deg, #ee4d2d, #c2410c);
    color: #fff;
    padding: 24px 16px 22px;
    text-align: center;
  }
  header.site h1 { margin: 0 0 6px; font-size: 24px; }
  header.site p { margin: 0; opacity: .85; font-size: 13px; }
  header.site .back {
    display: inline-block;
    color: #fff;
    text-decoration: none;
    font-size: 13px;
    opacity: .92;
    margin-bottom: 10px;
  }
  nav.toc {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 8px;
    margin: 16px 0 20px;
  }
  .nav-item {
    display: flex;
    gap: 10px;
    align-items: baseline;
    padding: 10px 12px;
    border-radius: 10px;
    text-decoration: none;
    color: var(--text);
    font-size: 14px;
  }
  .nav-item:active { background: var(--code-bg); }
  .nav-date { color: var(--accent); font-variant-numeric: tabular-nums; font-size: 12px; white-space: nowrap; font-weight: 600; }
  article {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 18px 24px;
    margin-bottom: 20px;
  }
  article h1 { font-size: 20px; line-height: 1.4; border: none; margin-top: 0; }
  article h2 { font-size: 18px; margin-top: 1.6em; padding-bottom: .3em; border-bottom: 1px solid var(--border); }
  article h3 { font-size: 16px; margin-top: 1.4em; }
  h4 { font-size: 15px; }
  p { font-size: 15px; margin: .8em 0; }
  li { font-size: 15px; }
  blockquote {
    margin: 1em 0;
    padding: 10px 14px;
    border-left: 4px solid var(--accent);
    background: var(--code-bg);
    border-radius: 0 10px 10px 0;
    color: var(--muted);
  }
  blockquote p { margin: .3em 0; font-size: 14px; }
  code {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--code-bg);
    padding: 2px 6px;
    border-radius: 5px;
    font-size: 13px;
  }
  pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  pre code { background: none; padding: 0; font-size: 12.5px; line-height: 1.6; }
  /* ---------- 语法高亮（hljs token，浅色主题，GitHub 风格） ---------- */
  .hljs-comment, .hljs-quote { color: #6a737d; font-style: italic; }
  .hljs-keyword, .hljs-selector-tag, .hljs-doctag { color: #d73a49; }
  .hljs-string, .hljs-regexp { color: #032f62; }
  .hljs-number, .hljs-literal, .hljs-built_in { color: #005cc5; }
  .hljs-title, .hljs-title.function_, .hljs-title.class_ { color: #6f42c1; }
  .hljs-type, .hljs-class .hljs-title { color: #22863a; }
  .hljs-attr, .hljs-attribute, .hljs-property, .hljs-variable.language_ { color: #e36209; }
  .hljs-params { color: #24292e; }
  .hljs-meta, .hljs-meta .hljs-keyword { color: #6f42c1; }
  .hljs-symbol, .hljs-bullet { color: #e36209; }
  .hljs-emphasis { font-style: italic; }
  .hljs-strong { font-weight: 600; }
  @media (prefers-color-scheme: dark) {
    .hljs-comment, .hljs-quote { color: #8b949e; }
    .hljs-keyword, .hljs-selector-tag, .hljs-doctag { color: #ff7b72; }
    .hljs-string, .hljs-regexp { color: #a5d6ff; }
    .hljs-number, .hljs-literal, .hljs-built_in { color: #79c0ff; }
    .hljs-title, .hljs-title.function_, .hljs-title.class_ { color: #d2a8ff; }
    .hljs-type, .hljs-class .hljs-title { color: #ffa657; }
    .hljs-attr, .hljs-attribute, .hljs-property, .hljs-variable.language_ { color: #ffa657; }
    .hljs-params { color: #e6e8eb; }
    .hljs-meta, .hljs-meta .hljs-keyword { color: #d2a8ff; }
    .hljs-symbol, .hljs-bullet { color: #ffa657; }
  }
  table { display: block; overflow-x: auto; border-collapse: collapse; font-size: 13px; max-width: 100%; }
  th, td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; }
  th { background: var(--code-bg); }
  details {
    background: var(--code-bg);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 1em 0;
    font-size: 14px;
  }
  summary { cursor: pointer; font-weight: 600; }
  a { color: var(--accent); }
  .pager {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
  }
  .pager a {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 14px;
    text-decoration: none;
    color: var(--text);
    font-size: 13px;
    line-height: 1.5;
    min-width: 0;
  }
  .pager .dir { display: block; color: var(--muted); font-size: 12px; margin-bottom: 2px; }
  .pager .ellipsis { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
`;

// ---------- 目录页（永远轻量，只有列表） ----------
const listItems = articles
  .map(
    (a) =>
      `<a class="nav-item" href="${a.page}"><span class="nav-date">${a.meta.date}</span><span class="nav-title">${esc(a.meta.title)}${a.meta.subtitle ? " · " + esc(a.meta.subtitle) : ""}</span></a>`
  )
  .join("\n");

const indexPage = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>PyTorch 每日一课</title>
<style>${CSS}</style>
</head>
<body>
<header class="site">
  <h1>🔥 PyTorch 每日一课</h1>
  <p>每天一个领域 · 不求全覆盖，只求真懂 · 共 ${articles.length} 篇</p>
</header>
<div class="wrap">
  <nav class="toc">${listItems}</nav>
</div>
</body>
</html>`;

// ---------- 文章页（每篇一个文件，点开才加载） ----------
function buildPostPage(a, i) {
  const newer = articles[i - 1]; // 列表按新到旧，i-1 是更新的一篇
  const older = articles[i + 1]; // i+1 是更早的一篇

  const pager = [
    older
      ? `<a href="${older.page}"><span class="dir">← 更早一篇</span><span class="ellipsis">${esc(older.meta.title)}${older.meta.subtitle ? " · " + esc(older.meta.subtitle) : ""}</span></a>`
      : `<a href="../index.html"><span class="dir">← 已是最早一篇</span><span class="ellipsis">返回目录</span></a>`,
    newer
      ? `<a href="${newer.page}"><span class="dir">更新一篇 →</span><span class="ellipsis">${esc(newer.meta.title)}${newer.meta.subtitle ? " · " + esc(newer.meta.subtitle) : ""}</span></a>`
      : `<a href="../index.html"><span class="dir">已是最新一篇</span><span class="ellipsis">返回目录</span></a>`,
  ].join("\n");

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>${esc(a.meta.title)} · PyTorch 每日一课</title>
<style>${CSS}</style>
</head>
<body>
<header class="site">
  <a class="back" href="../index.html">← 返回目录</a>
  <h1>🔥 PyTorch 每日一课</h1>
  <p>每天一个领域 · 不求全覆盖，只求真懂</p>
</header>
<div class="wrap">
  <article>${a.html}</article>
  <nav class="pager">${pager}</nav>
</div>
</body>
</html>`;
}

// ---------- 写出 ----------
// 清空旧的构建产物，避免残留旧文件
fs.rmSync(OUT_DIR, { recursive: true, force: true });
fs.mkdirSync(POSTS_DIR, { recursive: true });

fs.writeFileSync(path.join(OUT_DIR, "index.html"), indexPage);
for (let i = 0; i < articles.length; i++) {
  fs.writeFileSync(
    path.join(OUT_DIR, articles[i].page),
    buildPostPage(articles[i], i)
  );
}

console.log(
  `Built site: index.html (TOC) + ${articles.length} post page(s) -> ${OUT_DIR}`
);
