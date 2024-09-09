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
    title: 'My Docs',
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
	  './src/tailwind.css',
	],
    social: {
      github: 'https://github.com/withastro/starlight',
      youtube: 'https://space.bilibili.com/3546706348084176',
    },
    sidebar: [{
      label: '开篇文档',
      slug: 'train/zero2hero'
    },{
      label: '电子电路设计',
      items: [{
        label: '前言',
        slug: 'electronics/index_ecd'
      }, {
        label: '电子学',
        autogenerate: {
          directory: 'electronics/TAofE'
        }
      }]
    }, {
      label: '万能工科生基础',
      items: [
      // Each item here is one entry in the navigation menu.
      {
        label: '1.实用工具',
        slug: 'train/1tools'
      }]
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