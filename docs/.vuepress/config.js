module.exports = {
  // 其他配置...
  head: [
    // 其他头部标签...
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css' }],
    ['script', { src: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js' }],
    ['script', { src: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js' }]
  ],
  plugins: [
    'vuepress-plugin-katex'
    // 或者使用 MathJax
    // '@vuepress/plugin-mathjax'
  ],
}