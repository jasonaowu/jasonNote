import { sidebar } from "vuepress-theme-hope";

// 图标：https://theme-hope.vuejs.press/zh/guide/interface/icon.html#%E8%AE%BE%E7%BD%AE%E5%9B%BE%E6%A0%87
// https://fontawesome.com/search?m=free&o=r
export default sidebar({
  "": [
    // "/DailyRoutine",
    // "/Fitness",
    // 读书笔记架构更换到 docsify，不能使用相对链接
    // { text: "读书笔记", icon: "fa6-brands:readme", link: "https://jasonaowu.github.io/jasonNote/reading/" },
    // 指定显示页面
    {
      text: "🔡 代码编程",
      icon: "",
      prefix: "/code/",
      collapsible: true,
      children: [
        "README.md",
        {
          text: "Basic",
          icon: "fa6-solid:cube",
          collapsible: true,
          children: ["Markdown.md", "Electron.md", "AutoHotkey.md", "Regex.md"],
        },
        {
          text: "FrondEnd",
          icon: "fa6-solid:object-group",
          collapsible: true,
          children: ["Vue.md", "HTML.md", "Javascript.md", "Python.md"],
        },
      ],
    },
    {
      text: "⚓ 机器学习",
      // icon: "fa6-solid:ticket-simple",
      icon: "",
      prefix: "/ML/",
      collapsible: true,
      children: [
        // "README.md",
        {
          text: "树",
          icon: "fa6-solid:tree",
          collapsible: true,
          children: ["XGBoost.md", "Cart.md"],
        },
        {
          text: "聚类",
          icon: "fa6-solid:border-all",
          collapsible: true,
          children: ["KNN.md"],
        },
      ],
    },
    {
      text: "🍉 深度学习",
      // icon: "fa6-solid:ticket-simple",
      icon: "",
      prefix: "/DL/",
      collapsible: true,
      children: [
        // "README.md",
        {
          text: "基础知识",
          icon: "fa6-solid:tree",
          collapsible: true,
          children: ["激活函数.md"],
        },
        {
          text: "CV",
          icon: "fa6-solid:image",
          collapsible: true,
          children: ["ResNet.md"],
        },
        {
          text: "NLP",
          icon: "fa6-solid:spell-check",
          collapsible: true,
          children: ["Word2Vec.md"],
        },
      ],
    },
    {
      text: "🍡 大模型",
      // icon: "fa6-solid:ticket-simple",
      icon: "",
      prefix: "/LLM/",
      collapsible: true,
      children: [
        // "README.md",
        {
          text: "微调",
          icon: "fa6-solid:bolt",
          collapsible: false,
          // children: ["LoRA.md", "SFT"],
          link: "LLM/微调/"
        },
        {
          text: "框架",
          icon: "fa6-solid:sliders",
          collapsible: true,
          children: ["RAG.md"], 
        },
      ],
    },
    {
      text: "🧰 软件应用",
      icon: "",
      prefix: "/apps/",
      link: "",
      collapsible: true,
      children: [
        // "Applist.md",
        "Chrome.md",
        "toolbox.md",
        // {
        //   text: "其他",
        //   icon: "fa6-solid:code-compare",
        //   collapsible: true,
        //   children: ["design.md"],
        // },
      ],
    },
    
    // {
    //   text: "🛖 生活记录",
    //   icon: "",
    //   prefix: "/family/",
    //   collapsible: true,
    //   children: "structure",
    // },
    // {
    //   text: "加密目录",
    //   icon: "material-symbols:encrypted",
    //   prefix: "/encrypt/",
    //   collapsible: true,
    //   children: "structure",
    // },
    {
      text: "博客文章",
      icon: "fa6-solid:feather-pointed",
      prefix: "/_posts/",
      link: "/blog",
      collapsible: true,
      children: "structure",
    },
  ],
  // 专题区（独立侧边栏）
  "/apps/topic/": "structure",
  // 如果你不想使用默认侧边栏，可以按照路径自行设置。但需要去掉下方配置中的注释，以避免博客和时间轴出现异常。_posts 目录可以不存在。
  /*"/_posts/": [
    {
      text: "博客文章",
      icon: "fa6-solid:feather-pointed",
      prefix: "",
      link: "/blog",
      collapsible: true,
      children: "structure",
    },
  ], */
});
