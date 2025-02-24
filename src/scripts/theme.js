// src/scripts/giscus-theme.js
function updateGiscusTheme() {
    const frame = document.querySelector('iframe.giscus-frame');
    if (frame) {
      // 从 localStorage 获取当前主题
      const currentTheme = localStorage.getItem('starlight-theme') || 'auto';
      // 根据主题设置对应的 giscus 主题
      const giscusTheme = currentTheme === 'dark' ? 'dark_dimmed' : 'light';
      
      frame.contentWindow.postMessage(
        {
          giscus: {
            setConfig: {
              theme: giscusTheme
            }
          }
        },
        'https://giscus.app'
      );
    }
  }
  
  // 监听主题选择器变化
  document.addEventListener('DOMContentLoaded', function() {
    const themeSelect = document.querySelector('starlight-theme-select');
    if (themeSelect) {
      themeSelect.addEventListener('change', updateGiscusTheme);
    }
  });
  
  // 监听 localStorage 变化
  window.addEventListener('storage', (e) => {
    if (e.key === 'starlight-theme') {
      updateGiscusTheme();
    }
  });
  
  // 初始化时执行一次
  document.addEventListener('DOMContentLoaded', updateGiscusTheme);
  