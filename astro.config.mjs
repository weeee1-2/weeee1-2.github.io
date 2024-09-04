import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightImageZoom from 'starlight-image-zoom';
import remarkMath from "remark-math";
import rehypeMathjax from 'rehype-mathjax';
import remarkEmbedImages from 'remark-embed-images';

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
      github: 'https://github.com/withastro/starlight'
    },
    sidebar: [{
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
      label: 'Guides',
      items: [
      // Each item here is one entry in the navigation menu.
      {
        label: 'Example Guide',
        slug: 'guides/example'
      }]
    }, {
      label: 'Reference',
      autogenerate: {
        directory: 'reference'
      }
    }]
  }), 
  tailwind({
	// 禁用默认的基础样式
	applyBaseStyles: false,
  })],
  markdown: {
    // 应用于 .md 和 .mdx 文件
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeMathjax, remarkEmbedImages]
  }
});