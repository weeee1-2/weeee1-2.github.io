import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightImageZoom from 'starlight-image-zoom';
import remarkMath from "remark-math";
import rehypeMathjax from 'rehype-mathjax';

import tailwind from "@astrojs/tailwind";

// https://astro.build/config
export default defineConfig({
  site: 'https://maindraster.github.io',
  base: "/",

  integrations: [starlight({
    plugins: [starlightImageZoom()],
    title: '🦄&🐟',
    locales: {
      root: {
        label: '简体中文',
        lang: 'zh-CN'
      }
    },
    customCss: [
    // 你的自定义 CSS 文件的相对路径
    './src/styles/root.css', 
	  './src/styles/search.css', 
    './src/styles/iconfont.css', 
	  './src/tailwind.css',
    './src/styles/picsize.css',
	],
    social: {
      github: 'https://github.com/maindraster/maindraster.github.io',
    },
    sidebar: [{
      label: '开篇文档',
      slug: 'zero2hero'
    },{
      label: '万能工科生基础',
      autogenerate: {
          directory: 'train'
        }
    },{
      label: '电子电路设计篇',
      slug: 'electronics/index_ecd'
    },{
      label: '嵌入式开发篇',
      slug: 'embed/index_emb'
    },{
      label: '人工智能篇',
      slug: 'ai/index_ai'
    }],
    lastUpdated: true,
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
  }
});