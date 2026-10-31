//! arXiv fetching and parsing.
//!
//! Fetches the arXiv Atom feed for the configured categories, enforces the
//! network allowlist (HTTPS only, `export.arxiv.org` / `arxiv.org`), rate
//! limits requests to at least one every 3 seconds per arXiv etiquette, and
//! parses entries into [`Paper`] structs. Feed parsing is pure and unit
//! tested against a saved fixture; HTTP goes through the [`FeedFetcher`]
//! trait so tests never touch the real arXiv.

// TODO(wiring): remove once `main` starts using this module; until then
// these types are only consumed by tests.
#![allow(dead_code)]

use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::redirect;
use thiserror::Error;
use tokio::sync::Mutex;
use url::Url;

/// Hosts we are allowed to talk to for paper metadata (all over HTTPS).
const ALLOWED_HOSTS: [&str; 2] = ["export.arxiv.org", "arxiv.org"];

/// Maximum accepted response body (5 MB).
const MAX_BODY_BYTES: usize = 5 * 1024 * 1024;

/// Maximum redirects followed (each re-checked against the allowlist).
const MAX_REDIRECTS: usize = 3;

/// Minimum spacing between arXiv requests (arXiv etiquette asks for >= 3s).
pub const MIN_REQUEST_INTERVAL: Duration = Duration::from_secs(3);

/// Errors from fetching or parsing arXiv data.
#[derive(Debug, Error)]
pub enum ArxivError {
    /// The URL is not HTTPS or its host is not on the allowlist.
    #[error("host not allowed: {0} (allowed: export.arxiv.org, arxiv.org)")]
    HostNotAllowed(String),
    /// A malformed URL.
    #[error("invalid URL {url}: {source}")]
    InvalidUrl {
        url: String,
        #[source]
        source: url::ParseError,
    },
    /// The HTTP request failed (network, timeout, status, size limit).
    #[error("arXiv request to {url} failed: {message}")]
    Request { url: String, message: String },
    /// The response body was not a parseable Atom feed.
    #[error("failed to parse arXiv feed: {0}")]
    Parse(#[from] feed_rs::parser::ParseFeedError),
}

impl From<reqwest::Error> for ArxivError {
    fn from(e: reqwest::Error) -> Self {
        ArxivError::Request {
            url: e.url().map(|u| u.as_str().to_string()).unwrap_or_default(),
            message: e.to_string(),
        }
    }
}

/// One arXiv paper as extracted from the Atom feed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paper {
    /// arXiv id, e.g. "2401.12345v1" or "cs/0501001v1".
    pub id: String,
    /// Stable abs page URL, e.g. "https://arxiv.org/abs/2401.12345".
    pub url: String,
    /// Paper title (whitespace-normalised).
    pub title: String,
    /// Abstract text (whitespace-normalised).
    pub summary: String,
    /// Authors in feed order.
    pub authors: Vec<String>,
    /// Primary category, e.g. "cs.CL".
    pub primary_category: Option<String>,
    /// All categories (primary first, deduplicated).
    pub categories: Vec<String>,
    /// When the entry was published, if present.
    pub published: Option<DateTime<Utc>>,
}

impl Paper {
    /// The version-less arXiv id, e.g. "2501.00045" from "2501.00045v2".
    pub fn id_no_version(&self) -> &str {
        strip_version_suffix(&self.id)
    }

    /// Title + abstract, used by the keyword prefilter.
    pub fn text_for_filter(&self) -> String {
        format!("{}\n{}", self.title, self.summary)
    }
}

/// Strips a trailing `v<digits>` version suffix, e.g. "2501.00045v2" ->
/// "2501.00045". Returns the input unchanged when there is no such suffix.
fn strip_version_suffix(id: &str) -> &str {
    if let Some(pos) = id.rfind('v') {
        let (head, tail) = id.split_at(pos);
        if !head.is_empty() && tail.len() > 1 && tail[1..].bytes().all(|b| b.is_ascii_digit()) {
            return head;
        }
    }
    id
}

/// Percent-encodes a category term for use in an arXiv query.
///
/// arXiv categories are plain tokens (`cs.CL`, `math.AG`, `cs.DC`); anything
/// outside the unreserved set is encoded defensively.
fn encode_category(term: &str) -> String {
    let mut out = String::with_capacity(term.len());
    for b in term.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Builds the arXiv Atom query URL for the given categories, sorted by
/// submission date (newest first).
pub fn build_query_url(categories: &[String]) -> String {
    // Built by hand: arXiv expects literal `+` as the space between
    // OR-grouped terms, which `Url::parse_with_params` would encode as %2B.
    let cats = categories
        .iter()
        .map(|c| format!("cat:{}", encode_category(c)))
        .collect::<Vec<_>>()
        .join("+OR+");
    format!(
        "https://export.arxiv.org/api/query?search_query={cats}\
         &sortBy=submittedDate&sortOrder=descending&max_results=100"
    )
}

/// Validates that a URL is HTTPS and its host is on the allowlist.
pub fn check_url_allowed(url_str: &str) -> Result<Url, ArxivError> {
    let url = Url::parse(url_str).map_err(|source| ArxivError::InvalidUrl {
        url: url_str.to_string(),
        source,
    })?;
    let host = url.host_str().unwrap_or("");
    if url.scheme() == "https" && ALLOWED_HOSTS.contains(&host) {
        Ok(url)
    } else {
        Err(host_not_allowed_error(host))
    }
}

fn host_not_allowed_error(host: &str) -> ArxivError {
    ArxivError::HostNotAllowed(host.to_string())
}

/// Parses an arXiv Atom feed body into papers. Pure function; skips entries
/// it cannot make sense of rather than failing the whole feed.
pub fn parse_atom_feed(body: &str) -> Result<Vec<Paper>, ArxivError> {
    let feed = feed_rs::parser::parse(body.as_bytes())?;
    Ok(feed.entries.iter().filter_map(entry_to_paper).collect())
}

/// Converts one feed entry into a [`Paper`], or `None` if it lacks a usable
/// id/title/summary.
fn entry_to_paper(entry: &feed_rs::model::Entry) -> Option<Paper> {
    // The id is a full URL like https://arxiv.org/abs/2401.12345v1.
    let short_id = entry.id.rsplit('/').next()?.to_string();
    if short_id.is_empty() {
        return None;
    }

    let title = normalise_ws(&entry.title.as_ref()?.content);
    if title.is_empty() {
        return None;
    }
    let summary = normalise_ws(&entry.summary.as_ref()?.content);
    if summary.is_empty() {
        return None;
    }

    // Canonical https abs page (version-less), derived from the id rather
    // than trusting feed links, which arXiv serves as http:// with versions.
    let url = format!("https://arxiv.org/abs/{}", strip_version_suffix(&short_id));

    let mut categories: Vec<String> = Vec::new();
    for category in &entry.categories {
        if !categories.contains(&category.term) {
            categories.push(category.term.clone());
        }
    }
    // arXiv puts the primary category first when present.
    let primary_category = categories.first().cloned();

    Some(Paper {
        id: short_id,
        url,
        title,
        summary,
        authors: entry
            .authors
            .iter()
            .filter_map(|a| a.name.clone())
            .collect(),
        primary_category,
        categories,
        published: entry.published,
    })
}

/// Collapses all whitespace runs into single spaces and trims.
fn normalise_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Abstracts one HTTP GET of a feed body, for faking in tests.
#[async_trait::async_trait]
pub trait FeedFetcher: Send + Sync {
    /// Fetches the body at `url`, enforcing the allowlist, timeouts, the
    /// redirect cap with host re-checks, and the response size limit.
    async fn fetch(&self, url: &str) -> Result<String, ArxivError>;
}

/// Rate limiter that spaces arXiv requests at least
/// [`MIN_REQUEST_INTERVAL`] apart. Cheap to clone: clones share state.
#[derive(Debug, Clone, Default)]
pub struct RateLimiter {
    last_request: std::sync::Arc<Mutex<Option<std::time::Instant>>>,
    interval: Duration,
}

impl RateLimiter {
    /// Creates a limiter with the given minimum spacing.
    pub fn new(interval: Duration) -> Self {
        Self {
            last_request: std::sync::Arc::new(Mutex::new(None)),
            interval,
        }
    }

    /// Waits until at least `interval` has passed since the last call.
    pub async fn wait(&self) {
        let mut last = self.last_request.lock().await;
        match *last {
            Some(prev) => {
                let elapsed = prev.elapsed();
                if elapsed < self.interval {
                    tokio::time::sleep(self.interval - elapsed).await;
                }
            }
            None => tokio::time::sleep(Duration::ZERO).await,
        }
        *last = Some(std::time::Instant::now());
    }
}

/// Real HTTP fetcher built on reqwest.
pub struct ReqwestFetcher {
    client: reqwest::Client,
}

impl ReqwestFetcher {
    /// Creates a fetcher with the given contact email for the User-Agent.
    pub fn new(contact_email: &str) -> Result<Self, ArxivError> {
        // At most MAX_REDIRECTS hops; every hop must stay on HTTPS and on an
        // allowed host. `previous()` includes the original URL, so a chain of
        // N redirects has N+1 entries there.
        let policy = redirect::Policy::custom(move |attempt| {
            if attempt.previous().len() > MAX_REDIRECTS {
                attempt.error("too many redirects")
            } else if attempt
                .url()
                .host_str()
                .is_some_and(|h| ALLOWED_HOSTS.contains(&h))
                && attempt.url().scheme() == "https"
            {
                attempt.follow()
            } else {
                attempt.stop()
            }
        });
        let client = reqwest::Client::builder()
            .redirect(policy)
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .user_agent(format!(
                "paper-radar/0.1.0 (https://github.com/soketlabs/paper-radar; mailto:{contact_email})"
            ))
            .build()?;
        Ok(Self { client })
    }
}

#[async_trait::async_trait]
impl FeedFetcher for ReqwestFetcher {
    async fn fetch(&self, url_str: &str) -> Result<String, ArxivError> {
        let url = check_url_allowed(url_str)?;
        let mut response = self
            .client
            .get(url)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| ArxivError::Request {
                url: url_str.to_string(),
                message: e.to_string(),
            })?;

        // Enforce the size cap even when Content-Length lies or is absent.
        if let Some(len) = response.content_length() {
            if len as usize > MAX_BODY_BYTES {
                return Err(too_large_error(url_str));
            }
        }
        let mut body: Vec<u8> = Vec::with_capacity(4096);
        while let Some(chunk) = response.chunk().await? {
            body.extend_from_slice(&chunk);
            if body.len() > MAX_BODY_BYTES {
                return Err(too_large_error(url_str));
            }
        }
        String::from_utf8(body).map_err(|_| ArxivError::Request {
            url: url_str.to_string(),
            message: "response body is not valid UTF-8".to_string(),
        })
    }
}

fn too_large_error(url: &str) -> ArxivError {
    ArxivError::Request {
        url: url.to_string(),
        message: format!("response exceeds {} bytes", MAX_BODY_BYTES),
    }
}

/// Reads a fixture file from `src/arxiv/fixtures/`.
#[cfg(test)]
pub(crate) fn fixture_path(name: &str) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("src/arxiv/fixtures");
    p.push(name);
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fetcher that never performs I/O: it checks the URL against the same
    /// allowlist logic and returns a canned body.
    struct FakeFetcher {
        bodies: Vec<(String, String)>,
        requested: Mutex<Vec<String>>,
    }

    impl FakeFetcher {
        fn new(bodies: Vec<(String, String)>) -> Self {
            Self {
                bodies,
                requested: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl FeedFetcher for FakeFetcher {
        async fn fetch(&self, url: &str) -> Result<String, ArxivError> {
            check_url_allowed(url)?;
            self.requested.lock().await.push(url.to_string());
            for (prefix, body) in &self.bodies {
                if url.starts_with(prefix) {
                    return Ok(body.clone());
                }
            }
            Err(ArxivError::Request {
                url: url.to_string(),
                message: "no canned body".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn fetch_rejects_disallowed_hosts_and_accepts_arxiv() {
        let fetcher = FakeFetcher::new(vec![(
            "https://export.arxiv.org/api/query".into(),
            "<feed/>".into(),
        )]);

        // Allowed.
        fetcher
            .fetch("https://export.arxiv.org/api/query?search_query=cat:cs.CL")
            .await
            .expect("export.arxiv.org should be allowed");

        // Wrong host.
        let err = fetcher
            .fetch("https://evil.example.com/feed")
            .await
            .expect_err("non-arXiv host must be rejected");
        assert!(
            err.to_string().contains("not allowed"),
            "should say host not allowed: {err}"
        );

        // Not HTTPS.
        let err = fetcher
            .fetch("http://export.arxiv.org/api/query")
            .await
            .expect_err("plain HTTP must be rejected");
        assert!(
            err.to_string().contains("not allowed"),
            "should reject non-HTTPS: {err}"
        );

        // The rejected URLs must never have been "requested".
        let requested = fetcher.requested.lock().await;
        assert_eq!(requested.len(), 1, "only the allowed URL is fetched");
    }

    #[test]
    fn parses_fixture_atom_feed_into_papers() {
        let xml =
            std::fs::read_to_string(fixture_path("sample_feed.xml")).expect("fixture must exist");
        let papers = parse_atom_feed(&xml).expect("fixture must parse");

        assert_eq!(papers.len(), 2, "fixture has two entries");

        let p = &papers[0];
        assert_eq!(p.id, "2501.00001v1");
        assert_eq!(p.id_no_version(), "2501.00001");
        assert_eq!(p.url, "https://arxiv.org/abs/2501.00001");
        assert_eq!(p.title, "Speculative Decoding for Low-Resource Languages");
        assert!(p.summary.starts_with("We speed up inference"));
        assert_eq!(p.authors, vec!["Asha Rao", "Dev Patel"]);
        assert_eq!(p.primary_category.as_deref(), Some("cs.CL"));
        assert_eq!(p.categories, vec!["cs.CL", "cs.AI"]);
        assert!(p.published.is_some());

        let q = &papers[1];
        assert_eq!(q.id, "2501.00002v1");
        assert_eq!(q.primary_category.as_deref(), Some("cs.LG"));

        // Filter text combines title and abstract.
        let text = p.text_for_filter();
        assert!(text.contains("Speculative Decoding"));
        assert!(text.contains("We speed up inference"));
    }

    #[test]
    fn build_query_url_targets_arxiv_with_categories() {
        let url = build_query_url(&["cs.CL".to_string(), "cs.LG".to_string()]);
        assert!(url.starts_with("https://export.arxiv.org/api/query?"));
        // arXiv's documented query form: search_query=cat:cs.CL+OR+cat:cs.LG
        assert!(url.contains("search_query=cat:cs.CL+OR+cat:cs.LG&"));
        assert!(url.contains("sortBy=submittedDate"));
        assert!(url.contains("sortOrder=descending"));

        // The produced URL must itself pass the allowlist check.
        check_url_allowed(&url).expect("query URL must be allowed");
    }

    #[tokio::test]
    async fn rate_limiter_spaces_consecutive_calls() {
        let limiter = RateLimiter::new(Duration::from_millis(50));
        let start = std::time::Instant::now();
        limiter.wait().await;
        limiter.wait().await;
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(45),
            "second call should wait for the interval, took {elapsed:?}"
        );
    }
}
