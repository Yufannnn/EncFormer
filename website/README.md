# EncFormer interactive research website

The static website is hosted at https://yufannnn.github.io/EncFormer/ from
`main/docs` using GitHub Pages. It requires no build step.

The interactive trace illustrates the BERT encoder graph. Layer 1 shows QKV,
score, MBMax, value/output, residual LayerNorm, and FFN/GELU operations with
four secure conversion pairs. Remaining layers fast-forward to client-side
decryption. The browser does not run cryptographic inference or measure latency.
Reported benchmarks are transcribed from the manuscript, not browser timings.

The acknowledgements page includes the final production acknowledgement,
NUS Computing and A*STAR affiliations, and paper, code, and artifact links.
Institutional logos retain their original proportions and colors.

Preview from the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Verify the trace and optional analytics backend:

```bash
node --test website/tests/trace.test.mjs website/visitor-worker/test/worker.test.js
```

Analytics is disabled by default. The small visitor atlas displays explicitly
labelled sample data until a real endpoint is configured. See [DEPLOY.md](DEPLOY.md)
for the optional Cloudflare Worker/D1 setup. Never place API secrets in `docs/config.js`.

## Logo sources

- [NUS Computing](https://www.comp.nus.edu.sg/wp-content/uploads/2023/09/nus-computing-logo-1.png)
- [A*STAR](https://www.a-star.edu.sg/api/media/53d6d556-a0aa-4bcb-9ba8-4eb1fc9ae750/ASTAR%20HQ%20Corporate%20Website.png)
- EncFormer logo supplied by the project author.
