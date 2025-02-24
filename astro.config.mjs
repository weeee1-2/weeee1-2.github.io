import { defineConfig,passthroughImageService } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightImageZoom from 'starlight-image-zoom';
import remarkMath from "remark-math";
import rehypeMathjax from 'rehype-mathjax';
import starlightBlog from 'starlight-blog'
import starlightGiscus from 'starlight-giscus'

import tailwind from "@astrojs/tailwind";
import vercel from '@astrojs/vercel';

// https://astro.build/config
export default defineConfig({
  site: 'https://maindraster.netlify.app',
  base: "/",

  image: {
    service: passthroughImageService()
  },

  integrations: [starlight({
    plugins: [
      starlightGiscus({
        repo: 'maindraster/docgiscus',
        repoId: 'R_kgDON-oOVQ',
        category:'Q&A',
        categoryId:'DIC_kwDON-oOVc4CnRog',
        theme:'catppuccin_latte',
        lazy: true
    }),
      starlightBlog({
      title: "博客",
      postCount: 5,
      recentPostCount: 10,
    }),starlightImageZoom(),
    // starlightUtils({
    //   navLinks: {
    //   leading: { useSidebarLabelled:  "leading"  } ,
    // }})
    ],
    title: '🦄&🐟',
    tableOfContents: { minHeadingLevel: 2,
       maxHeadingLevel: 4
       },
    locales: {
      root: {
        label: '简体中文',
        lang: 'zh-CN'
      }
    },
    customCss: [
    './src/tailwind2.css',
    // 你的自定义 CSS 文件的相对路径
    './src/styles/root.css', 
      './src/styles/search.css', 
    './src/styles/iconfont.css', 
    './src/styles/picsize.css',
    ],
    social: {
      github: 'https://github.com/maindraster/maindraster.github.io',
      youtube: 'https://space.bilibili.com/3546706348084176'
    },
    components: {
      Header: "./src/components/Myheader.astro",
      MarkdownContent: "./src/components/MarkdownContent.astro",
    },
    sidebar: [{
      label: '开篇文档',
      slug: 'zero2hero'
    },{
      label: '万能工科生教程',
      slug: 'train/tr_index'
    },{
      label: '电子电路设计篇',
      slug: 'electronics/index_ecd'
    },{
      label: '嵌入式开发篇',
      items: [{
        label: 'ESP32篇',
        autogenerate: {
          directory: 'embed/esp'
        }
      }]
    },{
      label: '人工智能篇',
      slug: 'ai/index_ai'
    },{
      label: '项目实战篇',
      items: [{
        label: '1.从非门到俄罗斯方块',
        slug: 'project/nand2tetris/nand2tetris'
      },{
        label: '2.一生一芯',
        slug: 'project/ysyx/ysyx'
      }]
    },
    ],
  }), 
  tailwind({
    // 禁用默认的基础样式
    applyBaseStyles: false,
  })],

  markdown: {
    // 应用于 .md 和 .mdx 文件
    smartypants: false,
    remarkPlugins: [remarkMath],
    rehypePlugins: [ rehypeMathjax],
    remarkRehype: { footnoteLabel: '参考', footnoteBackLabel: '返回正文' },
  },
  output: 'server',
  adapter: vercel()
});

// <script src="https://giscus.app/client.js"
//         data-repo="maindraster/docgiscus"
//         data-repo-id="R_kgDON-oOVQ"
//         data-category="Announcements"
//         data-category-id="DIC_kwDON-oOVc4CnRoe"
//         data-mapping="title"
//         data-strict="0"
//         data-reactions-enabled="1"
//         data-emit-metadata="0"
//         data-input-position="bottom"
//         data-theme="light_tritanopia"
//         data-lang="zh-CN"
//         crossorigin="anonymous"
//         async>
// </script>