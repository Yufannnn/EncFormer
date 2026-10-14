/* Shared controls for the standalone acknowledgements page. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  try {
    if (localStorage.getItem('encformer-theme') === 'light') document.documentElement.dataset.theme = 'light';
  } catch (_) {}
  $('themeToggle').addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('encformer-theme', theme); } catch (_) {}
  });
  $('menuToggle').addEventListener('click', () => {
    $('mobileNav').hidden = !$('mobileNav').hidden;
    $('menuToggle').setAttribute('aria-expanded', String(!$('mobileNav').hidden));
  });
  document.querySelectorAll('.mobile-nav a').forEach((a) => a.addEventListener('click', () => {
    $('mobileNav').hidden = true;
    $('menuToggle').setAttribute('aria-expanded', 'false');
  }));
  let toastTimer;
  function toast(text) {
    $('toast').textContent = text;
    $('toast').classList.add('is-visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.remove('is-visible'), 2900);
  }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); toast('Copied to clipboard'); }
    catch (_) { toast('Clipboard unavailable; select the text to copy it.'); }
  }
  $('copyAcknowledgement').addEventListener('click', () => copy($('acknowledgementText').textContent));
  $('copyCitation').addEventListener('click', () => copy($('bibtex').textContent));
  $('shareSite').addEventListener('click', async () => {
    const url = location.origin + location.pathname;
    if (navigator.share) {
      try { await navigator.share({title:'EncFormer · Acknowledgements & Resources', url}); return; }
      catch (error) { if (error.name === 'AbortError') return; }
    }
    await copy(url);
  });
  window.addEventListener('scroll', () => {
    const height = document.documentElement.scrollHeight - window.innerHeight;
    $('readProgress').style.width = `${height > 0 ? Math.min(100, window.scrollY / height * 100) : 0}%`;
  }, {passive:true});
})();
