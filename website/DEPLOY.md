# Website deployment — GitHub Pages + optional visitor analytics

The main website requires **no build tools, API keys or server**. The optional public visitor atlas requires a Cloudflare account and Worker/D1 database. The site remains hosted entirely on GitHub Pages.

## A. Publish in your existing EncFormer repository (recommended)

1. Extract this ZIP. Copy the entire **`docs/`** directory into the root of `Yufannnn/EncFormer`. The final path must contain `docs/index.html` and `docs/.nojekyll`.
2. Commit and push `docs/` to `main` in the existing repository. Do not overwrite your runtime code or release assets.
3. In the repository, open **Settings → Pages**.
4. Under **Build and deployment**, select **Deploy from a branch**. Select branch **`main`**, folder **`/docs`**, and click **Save**.
5. After the GitHub Pages deployment completes, visit:
   `https://yufannnn.github.io/EncFormer/`

If GitHub Pages is already configured with another publishing source, coordinate the change with your existing site. GitHub Pages' URL path is case-sensitive on a project site. All local CSS/JS/assets use relative URLs so they work under `/EncFormer/`.

**Local preview before publishing:**

```bash
python3 -m http.server 8000 --directory docs
# open http://localhost:8000
```

For a standalone repo instead of `EncFormer`, put the contents of `docs/` at the publishing root or choose an Actions-based Pages workflow and update canonical URLs.

## B. Enable public visitor counts + country-level reader map (optional)

The site initially displays an explicitly labeled **sample preview** of the visitor atlas; it does **not** pretend those visits are real. To replace it with live aggregate counts:

1. Create/log into a Cloudflare account, and install Node.js 20+ locally. In the extracted package:

   ```bash
   cd website/visitor-worker
   npm install
   npx wrangler login
   npx wrangler d1 create encformer-visitors
   ```

2. Copy `wrangler.toml.example` to `wrangler.toml`. Copy the **`database_id`** printed by the D1 creation command into `wrangler.toml`. Keep `ALLOWED_ORIGINS = "https://yufannnn.github.io"` and `SITE_PATH = "/EncFormer/"` for the existing GitHub Pages repo.

3. Initialize the database, then deploy the Worker:

   ```bash
   npx wrangler d1 execute encformer-visitors --remote --file schema.sql
   npx wrangler deploy
   ```

4. Wrangler prints a public HTTPS Worker URL, similar to `https://encformer-visitors.your-subdomain.workers.dev`. This is an **example**, not a preconfigured address.

5. Open `docs/config.js` and set:

   ```js
   window.ENCFORMER_CONFIG = {
     visitorApi: "https://encformer-visitors.your-subdomain.workers.dev",
     cloudflareAnalyticsToken: "",
   };
   ```

   Replace the example hostname with **your actual deployed Worker URL**. Commit and push the updated file to GitHub Pages. The site will now fetch `GET /api/summary` and issue one `POST /api/visit` per tab session, with no IP or fingerprint stored in D1.

6. Verify by opening the published site in a regular (non-bot) browser, refreshing the visitor panel, and checking whether the mode indicator reads **LIVE AGGREGATE DATA**. Counts begin at zero and only accumulate after the Worker is enabled. There is **no transfer of the preview numbers** to production.

**Local Worker checks:**

```bash
cd website/visitor-worker
npm test                 # Node's built-in test runner; no Wrangler install needed
npx wrangler dev         # optional: test the D1 Worker locally
```

For local browser testing with a `localhost` page, append the local origin to `ALLOWED_ORIGINS` and set `SITE_PATH = "/"` in the local Wrangler config. Do not accidentally deploy local test paths to production. If you use a custom site domain, add that exact HTTPS origin to `ALLOWED_ORIGINS`.

**Important:** Public view counters are approximate. Without server-side identity tracking or an authenticated visitor mechanism, you cannot guarantee one count per human visitor. This implementation intentionally does not attempt IP fingerprinting, and Cloudflare may see request IPs as part of delivering the Worker, even though the site never writes them into its database.

## C. Optional: private traffic analytics dashboard

For additional statistics like traffic sources, browser/device types, Core Web Vitals and country breakdowns, use [Cloudflare Web Analytics](https://developers.cloudflare.com/web-analytics/get-started/) for your GitHub Pages hostname. It is separate from the public visitor map:

1. Add `yufannnn.github.io` (or your custom domain) as a website in Cloudflare Web Analytics.
2. Copy the site's **public analytics beacon token** (not a Cloudflare API secret).
3. In `docs/config.js`, set `cloudflareAnalyticsToken: "..."` and republish.
4. Explore your detailed metrics privately in the Cloudflare dashboard.

The private analytics dashboard and the public Worker counters may **not** report identical numbers, because tracking definitions, ad blockers, bot handling and privacy choices differ.

## D. Sharing and search visibility

- The social preview is `docs/assets/social-preview.png`; Open Graph metadata in `docs/index.html` already targets the suggested `yufannnn.github.io/EncFormer/` URL.
- Add the site URL to the repository **About → Website** field.
- Link the demo from the repository README and the paper/arXiv project page where appropriate.
- You can add a custom domain later without migrating the site; update the Worker `ALLOWED_ORIGINS` and Open Graph URLs.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| 404 after enabling Pages | Check `main /docs`, case-sensitive URL, and Pages build status. |
| Site loads without styles | Ensure `docs/styles.css` and `docs/app.js` were committed and not moved. |
| Visitor panel says preview | `docs/config.js` has no Worker URL, or the Worker returned an error. |
| Analytics CORS error | Add the exact website origin (no path) to `ALLOWED_ORIGINS`, then redeploy Worker. |
| POST visit returns 400 | `SITE_PATH` must match the browser's `location.pathname` (`/EncFormer/`). |
| POST visit returns 503 | Initialize D1 and confirm `database_id` and D1 `DB` binding in Wrangler config. |
| Map shows no country for some region | The included Natural Earth basemap covers major countries; small territories may not have a marker. |
| Private analytics differs from public count | Expected; definitions and tracking blockers differ. |

References: [GitHub Pages publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site), [Cloudflare Web Analytics setup](https://developers.cloudflare.com/web-analytics/get-started/).
