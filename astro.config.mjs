import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightImageZoom from 'starlight-image-zoom';

import remarkMath from "remark-math"
import rehypeMathjax from 'rehype-mathjax'

// https://astro.build/config
export default defineConfig({
	site: 'https://maindraster.github.io',
	base: "/",
	integrations: [
		starlight({
			plugins: [starlightImageZoom()],
			title: 'My Docs',
			locales: {
				root: {
				  label: '简体中文',
				  lang: 'zh-CN',
				},
			},
			social: {
				github: 'https://github.com/withastro/starlight',
			},
			sidebar: [
				{
					label: '电子电路设计',
					items:[
						{ label: '前言', slug: 'electronics/index_ecd' },
						{
							label: '电子学',
							autogenerate: { directory: 'electronics/TAofE' },
						},
					],
				},
				{
					label: 'Guides',
					items: [
						// Each item here is one entry in the navigation menu.
						{ label: 'Example Guide', slug: 'guides/example' },
					],
				},
				{
					label: 'Reference',
					autogenerate: { directory: 'reference' },
				},
			],
		}),
	],
	markdown: {
		// 应用于 .md 和 .mdx 文件
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeMathjax],
	},
});
