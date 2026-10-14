-- Privacy-first visitor counters: aggregate only, no IP address, cookies,
-- user agent, fingerprint, request IDs, or per-person data in storage.
CREATE TABLE IF NOT EXISTS daily_country_views (
  day TEXT NOT NULL,
  country TEXT NOT NULL,
  views INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(day, country)
);
CREATE INDEX IF NOT EXISTS idx_country_views_country ON daily_country_views(country);
