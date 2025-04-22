module.exports = {
  // 其他配置...
  head: [
    // 其他头部标签...
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css' }]
  ],
  markdown: {
    extendMarkdown: md => {
      md.use(require('markdown-it-katex'))
    }
  },
  // 可以移除 vuepress-plugin-katex 插件
  plugins: []
}