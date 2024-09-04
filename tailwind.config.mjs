import starlightPlugin from '@astrojs/starlight-tailwind';

// Generated color palettes
const accent = { 200: '#feb3a6', 600: '#c90e00', 900: '#640300', 950: '#460b05' };
const gray = { 100: '#f9f5f5', 200: '#f3ecea', 300: '#c8c0be', 400: '#978784', 500: '#635451', 700: '#423432', 800: '#302321', 900: '#1d1715' };

/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {
			colors: { accent, gray },
			fontFamily: {
				// 你喜欢的文本字体。Starlight 默认使用系统字体堆栈。
				'sans': ['"LXGW WenKai"',],
				// 你喜欢的代码字体。Starlight 默认使用系统等宽字体。
				mono: ['"IBM Plex Mono"'],
			},
		},
	},
	plugins: [starlightPlugin()],
};