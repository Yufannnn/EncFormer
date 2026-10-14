/**
 * Edit only the values below to enable external telemetry.
 * GitHub Pages remains the website host.
 *
 * Run the optional Cloudflare Worker (see ../website/visitor-worker and ../website/DEPLOY.md),
 * then paste its HTTPS origin below, without a trailing slash.
 * Empty string = no tracking / no live counters.
 */
window.ENCFORMER_CONFIG = {
  visitorApi: "https://encformer-visitors.encformer-visitors.workers.dev",
  // Optional: Cloudflare Web Analytics site token for a richer private dashboard.
  // Geographies, traffic sources, Core Web Vitals, etc. are shown in Cloudflare.
  cloudflareAnalyticsToken: "",
};
