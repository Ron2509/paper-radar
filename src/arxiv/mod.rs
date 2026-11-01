//! arXiv fetching and parsing.
//!
//! Daily fetching uses arXiv's RSS feed (`https://rss.arxiv.org/rss/<cats>`),
//! which lists exactly the papers announced today: only items with announce
//! type "new" or "cross" are kept (replacements are updated old papers). The
//! Atom search API code ([`build_query_url`], [`fetch_by_query`]) is kept
//! for later id-based lookups: it queries a `submittedDate` range and pages
//! through results with `start` / `max_results` until an empty page (hard
//! cap [`MAX_PAPERS`]). The network allowlist (HTTPS only,
//! `export.arxiv.org` / `arxiv.org` / `rss.arxiv.org`) is enforced on every
//! request and on every redirect hop, requests are rate limited to at least
//! one every 3 seconds per arXiv etiquette, and entries are parsed into
//! [`Paper`] structs. Feed parsing is pure and unit tested against saved
//! fixtures; HTTP goes through the [`FeedFetcher`] trait so tests never
//! touch the real arXiv.

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
const ALLOWED_HOSTS: [&str; 3] = ["export.arxiv.org", "arxiv.org", "rss.arxiv.org"];

/// Maximum accepted response body (5 MB).
const MAX_BODY_BYTES: usize = 5 * 1024 * 1024;

/// Maximum redirects followed (each re-checked against the allowlist).
const MAX_REDIRECTS: usize = 3;

/// Results requested per page (`max_results`).
pub const PAGE_SIZE: usize = 200;

/// Hard cap on the number of papers one fetch run may return.
pub const MAX_PAPERS: usize = 2000;

/// Minimum spacing between arXiv requests (arXiv etiquette asks for >= 3s).
pub const MIN_REQUEST_INTERVAL: Duration = Duration::from_secs(3);

/// Errors from fetching or parsing arXiv data.
#[derive(Debug, Error)]
pub enum ArxivError {
    /// The URL is not HTTPS or its host is not on the allowlist.
    #[error("host not allowed: {0} (allowed: export.arxiv.org, arxiv.org, rss.arxiv.org)")]
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
    /// The feed parsed as XML and contained items, but none of them matched
    /// the expected arXiv daily-listing shape (e.g. arXiv changed the RSS
    /// layout). An empty feed, or one holding only replacements, is not
    /// this error.
    #[error(
        "arXiv feed format was not recognised: {items} items found but none \
         could be parsed as an arXiv daily listing"
    )]
    UnrecognisedFeedFormat { items: usize },
    /// The requested submission window cannot be represented.
    #[error("invalid fetch window: {0}")]
    InvalidWindow(String),
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

/// Builds the arXiv Atom query URL for papers in the given categories
/// submitted between `from` and `to` (inclusive), sorted newest first,
/// returning the page starting at `start` (0-based) of up to `max_results`
/// entries.
///
/// The date filter uses the arXiv API's documented syntax
/// `submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]` with GMT times to the
/// minute. The URL is built by hand because arXiv expects a literal `+` as
/// the space both between OR-grouped terms and around `TO`, which
/// `Url::parse_with_params` would encode as `%2B`.
pub fn build_query_url(
    categories: &[String],
    from: DateTime<Utc>,
    to: DateTime<Utc>,
    start: usize,
    max_results: usize,
) -> String {
    let cats = categories
        .iter()
        .map(|c| format!("cat:{}", encode_category(c)))
        .collect::<Vec<_>>()
        .join("+OR+");
    format!(
        "https://export.arxiv.org/api/query?search_query=\
         submittedDate:[{}+TO+{}]+AND+({cats})\
         &sortBy=submittedDate&sortOrder=descending&start={start}&max_results={max_results}",
        format_gmt(from),
        format_gmt(to),
    )
}

/// Formats a timestamp as arXiv's `YYYYMMDDHHMM` (GMT, to the minute).
fn format_gmt(t: DateTime<Utc>) -> String {
    t.format("%Y%m%d%H%M").to_string()
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

/// What to do with one redirect hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RedirectDecision {
    /// The hop stays on an allowed HTTPS host; keep following.
    Follow,
    /// Refuse the hop, surfacing this message as an error.
    Reject(&'static str),
}

/// Decides whether a redirect hop to `url` is acceptable, given the URLs
/// visited so far (which includes the original request URL, like reqwest's
/// `Attempt::previous`). Shared by the real client's redirect policy and by
/// the fake fetcher's redirect simulation in tests.
fn redirect_decision(url: &Url, previous: &[Url]) -> RedirectDecision {
    if previous.len() > MAX_REDIRECTS {
        RedirectDecision::Reject("too many redirects")
    } else if url.scheme() == "https" && url.host_str().is_some_and(|h| ALLOWED_HOSTS.contains(&h))
    {
        RedirectDecision::Follow
    } else {
        RedirectDecision::Reject("redirect to disallowed host")
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

/// Builds the arXiv daily RSS feed URL for the given categories, e.g.
/// `https://rss.arxiv.org/rss/cs.CL+cs.LG`.
pub fn build_rss_url(categories: &[String]) -> String {
    let cats = categories
        .iter()
        .map(|c| encode_category(c))
        .collect::<Vec<_>>()
        .join("+");
    format!("https://rss.arxiv.org/rss/{cats}")
}

/// Announce type of an RSS item, from `<arxiv:announce_type>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnnounceType {
    /// New to arXiv ("new").
    New,
    /// Cross-listed into this category ("cross").
    Cross,
    /// Updated version of an existing paper ("replace", "replace-cross").
    Replace,
}

impl AnnounceType {
    /// Parses the announce type value; anything unknown counts as a
    /// replacement so we never surface stale papers as new.
    fn parse(value: &str) -> Self {
        match normalise_ws(value).as_str() {
            "new" => AnnounceType::New,
            "cross" => AnnounceType::Cross,
            _ => AnnounceType::Replace,
        }
    }

    /// Whether the item should go into the daily fetch.
    fn is_listable(&self) -> bool {
        matches!(self, AnnounceType::New | AnnounceType::Cross)
    }
}

/// The interesting parts of one RSS item description.
struct RssDescription {
    announce_type: AnnounceType,
    /// The abstract text with the prefix stripped (owned, since the prefix
    /// stripping may allocate).
    abstract_text: String,
}

/// Splits an arXiv RSS `<description>` of the documented form
/// `arXiv:<id> Announce Type: <type>\nAbstract: <text>` into the announce
/// type and the abstract text. Returns `None` if the shape is not
/// recognised (the item is then skipped rather than mis-parsed).
fn split_rss_description(description: &str) -> Option<RssDescription> {
    // feed-rs decodes HTML entities, but the raw feed sometimes still has
    // stray markup; strip it defensively before matching the prefix.
    let text = strip_html_tags(description);
    let after_arxiv_id = text.split_once(" Announce Type: ")?;
    let (rest, abstract_text) = after_arxiv_id
        .1
        .split_once("\nAbstract: ")
        .or_else(|| after_arxiv_id.1.split_once(" Abstract: "))?;
    if rest.contains('\n') || rest.is_empty() {
        return None;
    }
    Some(RssDescription {
        announce_type: AnnounceType::parse(rest),
        abstract_text: abstract_text.trim().to_string(),
    })
}

/// Removes `<...>` tags and collapses whitespace.
fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    normalise_ws(&out)
}

/// Converts one parsed RSS item into a [`Paper`] with the announce type.
///
/// The id comes from the guid (`oai:arXiv.org:<id>`, falling back to the
/// link); the abs URL is always rebuilt as canonical HTTPS from that id.
fn rss_entry_to_paper(entry: &feed_rs::model::Entry) -> Option<(Paper, AnnounceType)> {
    let description = entry.summary.as_ref()?;
    let parsed = split_rss_description(&description.content)?;

    // guid is "oai:arXiv.org:2501.00001v1"; fall back to the item link.
    let raw_id = entry
        .id
        .strip_prefix("oai:arXiv.org:")
        .unwrap_or(&entry.id)
        .rsplit('/')
        .next()
        .unwrap_or("");
    let raw_id = raw_id.trim();
    if raw_id.is_empty() {
        return None;
    }

    let title = normalise_ws(&entry.title.as_ref()?.content);
    if title.is_empty() {
        return None;
    }
    let summary = normalise_ws(&parsed.abstract_text);
    if summary.is_empty() {
        return None;
    }

    // Canonical https abs page (version-less), derived from the id rather
    // than trusting the link, which may be http or versioned.
    let url = format!("https://arxiv.org/abs/{}", strip_version_suffix(raw_id));

    let mut categories: Vec<String> = Vec::new();
    for category in &entry.categories {
        if !categories.contains(&category.term) {
            categories.push(category.term.clone());
        }
    }

    // dc:creator lists authors comma-separated; feed-rs puts each author
    // element in `authors` (one per item).
    let mut authors = Vec::new();
    for person in &entry.authors {
        let names = person.name.as_deref().unwrap_or("");
        for name in names.split(',') {
            let name = name.trim();
            if !name.is_empty() && !authors.iter().any(|a| a == name) {
                authors.push(name.to_string());
            }
        }
    }

    Some((
        Paper {
            id: raw_id.to_string(),
            url,
            title,
            summary,
            authors,
            primary_category: categories.first().cloned(),
            categories,
            published: entry.published,
        },
        parsed.announce_type,
    ))
}

/// Parses an arXiv daily RSS 2.0 feed body into the papers announced today,
/// keeping only "new" and "cross" listings (replacements are updated old
/// papers, not today's arrivals). Pure function.
///
/// An empty feed is fine (nothing was announced). A feed with items where
/// none parse — e.g. arXiv changed the description layout — is an error:
/// silently returning nothing would look like "no papers today" when really
/// we stopped understanding the feed. Items that parse but are filtered out
/// as replacements do not count as unparseable.
pub fn parse_rss_feed(body: &str) -> Result<Vec<Paper>, ArxivError> {
    let feed = feed_rs::parser::parse(body.as_bytes())?;
    if feed.entries.is_empty() {
        return Ok(Vec::new());
    }

    let mut any_parsed = false;
    let mut papers = Vec::new();
    for (paper, announce_type) in feed.entries.iter().filter_map(rss_entry_to_paper) {
        any_parsed = true;
        if announce_type.is_listable() {
            papers.push(paper);
        }
    }

    if !any_parsed {
        return Err(ArxivError::UnrecognisedFeedFormat {
            items: feed.entries.len(),
        });
    }
    Ok(papers)
}

/// Fetches today's announced papers for the given categories from the arXiv
/// daily RSS feed: one request, no paging.
pub async fn fetch_today(
    fetcher: &dyn FeedFetcher,
    limiter: &RateLimiter,
    categories: &[String],
) -> Result<Vec<Paper>, ArxivError> {
    let url = build_rss_url(categories);
    limiter.wait().await;
    let body = fetcher.fetch(&url).await?;
    let papers = parse_rss_feed(&body)?;
    tracing::debug!(count = papers.len(), "fetched today's arXiv papers via RSS");
    Ok(papers)
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

/// Fetches papers in the given categories submitted within `window` before
/// `now`, paging through the arXiv Atom search API by query.
///
/// Requests pages of [`PAGE_SIZE`] results via `start` / `max_results`,
/// waiting on `limiter` between pages (arXiv etiquette: >= 3s), and stops
/// at the first empty page. Never returns more than [`MAX_PAPERS`] papers.
/// Note: the search API only contains papers after they are announced, so
/// for daily digests prefer [`fetch_today`]; this is for query/id lookups.
pub async fn fetch_by_query(
    fetcher: &dyn FeedFetcher,
    limiter: &RateLimiter,
    categories: &[String],
    now: DateTime<Utc>,
    window: Duration,
) -> Result<Vec<Paper>, ArxivError> {
    let window =
        chrono::Duration::from_std(window).map_err(|e| ArxivError::InvalidWindow(e.to_string()))?;
    let from = now
        .checked_sub_signed(window)
        .ok_or_else(|| ArxivError::InvalidWindow("window start is out of range".to_string()))?;

    let mut papers: Vec<Paper> = Vec::new();
    let mut start = 0usize;
    while papers.len() < MAX_PAPERS {
        limiter.wait().await;
        let url = build_query_url(categories, from, now, start, PAGE_SIZE);
        let body = fetcher.fetch(&url).await?;
        let page = parse_atom_feed(&body)?;
        tracing::debug!(
            start,
            got = page.len(),
            total = papers.len(),
            "fetched arXiv page"
        );
        if page.is_empty() {
            break;
        }
        papers.extend(page);
        start += PAGE_SIZE;
    }
    papers.truncate(MAX_PAPERS);
    tracing::debug!(count = papers.len(), "arXiv fetch complete");
    Ok(papers)
}

/// Real HTTP fetcher built on reqwest.
pub struct ReqwestFetcher {
    client: reqwest::Client,
}

impl ReqwestFetcher {
    /// Creates a fetcher with the given contact email for the User-Agent.
    pub fn new(contact_email: &str) -> Result<Self, ArxivError> {
        // At most MAX_REDIRECTS hops; every hop must stay on HTTPS and on an
        // allowed host, otherwise the request fails loudly instead of
        // silently stopping at an off-allowlist response.
        let policy = redirect::Policy::custom(move |attempt| {
            match redirect_decision(attempt.url(), attempt.previous()) {
                RedirectDecision::Follow => attempt.follow(),
                RedirectDecision::Reject(message) => attempt.error(message),
            }
        });
        let client = reqwest::Client::builder()
            .redirect(policy)
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .user_agent(format!("paper-radar/0.1.0 (mailto:{contact_email})"))
            .build()
            .map_err(|e| ArxivError::Request {
                url: "client build".to_string(),
                message: e.to_string(),
            })?;
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
            .await
            .map_err(|e| request_error(url_str, &e))?
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
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| request_error(url_str, &e))?
        {
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

/// Maps a transport-level reqwest error (network, timeout, redirect policy,
/// ...) onto [`ArxivError::Request`], keeping the offending URL.
fn request_error(url_str: &str, e: &reqwest::Error) -> ArxivError {
    ArxivError::Request {
        url: url_str.to_string(),
        message: e.to_string(),
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
    use chrono::TimeZone as _;

    /// A fetcher that never performs I/O: it checks the URL against the same
    /// allowlist logic, simulates redirects through the same policy the real
    /// client uses, and returns a canned body.
    struct FakeFetcher {
        /// (needle, body): first entry whose needle is contained in the URL
        /// wins; an empty needle matches everything.
        bodies: Vec<(String, String)>,
        /// (needle, location): if the URL contains needle, respond with a
        /// redirect to `location` instead of a body.
        redirects: Vec<(String, String)>,
        requested: Mutex<Vec<String>>,
    }

    impl FakeFetcher {
        fn new(bodies: Vec<(String, String)>) -> Self {
            Self {
                bodies,
                redirects: Vec::new(),
                requested: Mutex::new(Vec::new()),
            }
        }

        fn with_redirects(mut self, redirects: Vec<(String, String)>) -> Self {
            self.redirects = redirects;
            self
        }

        fn redirect_for(&self, url: &Url) -> Option<String> {
            self.redirects
                .iter()
                .find(|(needle, _)| url.as_str().contains(needle))
                .map(|(_, location)| location.clone())
        }
    }

    #[async_trait::async_trait]
    impl FeedFetcher for FakeFetcher {
        async fn fetch(&self, url: &str) -> Result<String, ArxivError> {
            let mut current = check_url_allowed(url)?;
            self.requested.lock().await.push(current.to_string());
            loop {
                if let Some(location) = self.redirect_for(&current) {
                    // Same rules as the real client: every hop must stay on
                    // an allowed HTTPS host, within the redirect budget.
                    let next = Url::parse(&location).map_err(|source| ArxivError::InvalidUrl {
                        url: location.clone(),
                        source,
                    })?;
                    let visited: Vec<Url> = self
                        .requested
                        .lock()
                        .await
                        .iter()
                        .filter_map(|u| Url::parse(u).ok())
                        .collect();
                    match redirect_decision(&next, &visited) {
                        RedirectDecision::Follow => {
                            current = next;
                            self.requested.lock().await.push(current.to_string());
                        }
                        RedirectDecision::Reject(message) => {
                            return Err(ArxivError::Request {
                                url: next.to_string(),
                                message: message.to_string(),
                            });
                        }
                    }
                } else {
                    for (needle, body) in &self.bodies {
                        if current.as_str().contains(needle.as_str()) {
                            return Ok(body.clone());
                        }
                    }
                    return Err(ArxivError::Request {
                        url: current.to_string(),
                        message: "no canned body".to_string(),
                    });
                }
            }
        }
    }

    /// A minimal Atom feed with no entries (what arXiv returns past the end
    /// of a result set).
    fn empty_feed_xml() -> &'static str {
        r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"#
    }

    /// One valid Atom entry for the given arXiv id.
    fn entry_xml(id: &str) -> String {
        format!(
            "<entry><id>http://arxiv.org/abs/{id}</id>\
             <updated>2025-01-02T09:30:00Z</updated>\
             <published>2025-01-02T09:30:00Z</published>\
             <title>Paper {id}</title><summary>Abstract of paper {id}.</summary>\
             <author><name>A. Author</name></author></entry>"
        )
    }

    /// An Atom feed with `n` entries.
    fn feed_with_n_entries(n: usize) -> String {
        let entries: String = (0..n)
            .map(|i| entry_xml(&format!("2501.{i:05}v1")))
            .collect();
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>"#
        )
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
    fn parses_fixture_rss_feed_keeping_only_new_and_cross() {
        let xml = std::fs::read_to_string(fixture_path("sample_rss_feed.xml"))
            .expect("fixture must exist");
        let papers = parse_rss_feed(&xml).expect("fixture must parse");

        // "replace" is dropped: only the new and cross listings remain.
        assert_eq!(papers.len(), 2, "only new + cross items are kept");
        assert!(
            !papers.iter().any(|p| p.id.starts_with("2501.00003")),
            "replaced paper must be skipped"
        );

        let p = &papers[0];
        assert_eq!(p.id, "2501.00001v1");
        assert_eq!(p.id_no_version(), "2501.00001");
        assert_eq!(p.url, "https://arxiv.org/abs/2501.00001");
        assert_eq!(
            p.title,
            "Faster Speculative Decoding for Low-Resource Languages"
        );
        // The "arXiv:<id> Announce Type: ... Abstract:" prefix is stripped.
        assert!(
            p.summary.starts_with("We speed up inference"),
            "{}",
            p.summary
        );
        assert!(!p.summary.contains("Announce Type"));
        assert!(!p.summary.contains("Abstract:"));
        assert_eq!(p.authors, vec!["Asha Rao", "Dev Patel"]);
        assert_eq!(p.primary_category.as_deref(), Some("cs.CL"));
        assert_eq!(p.categories, vec!["cs.CL", "cs.AI"]);
        assert!(p.published.is_some());

        let q = &papers[1];
        assert_eq!(q.id, "2501.00002v1");
        assert_eq!(q.url, "https://arxiv.org/abs/2501.00002");
        assert_eq!(q.authors, vec!["Mei Lin"]);
        assert!(q.summary.starts_with("We prune KV cache"));
    }

    #[test]
    fn unparseable_items_error_but_empty_or_replace_only_feeds_do_not() {
        // Items exist but none match the expected description layout (e.g.
        // arXiv changed the RSS format): this must be a loud error, not a
        // silent "no papers today".
        let unrecognised = r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>cs.CL updates on arXiv.org</title>
    <item>
      <title>Some Paper</title>
      <link>https://arxiv.org/abs/2501.00099</link>
      <description>A description with no announce-type marker at all.</description>
      <guid isPermaLink="false">oai:arXiv.org:2501.00099v1</guid>
    </item>
    <item>
      <title>Another Paper</title>
      <link>https://arxiv.org/abs/2501.00098</link>
      <description>Also not the documented arXiv daily-listing shape.</description>
      <guid isPermaLink="false">oai:arXiv.org:2501.00098v1</guid>
    </item>
  </channel>
</rss>"#;
        let err = parse_rss_feed(unrecognised).expect_err("unparseable items must be an error");
        let msg = err.to_string();
        assert!(
            msg.contains("format was not recognised"),
            "should say the format was not recognised: {msg}"
        );
        assert!(
            msg.contains("2 items"),
            "should report how many items were found: {msg}"
        );

        // A feed holding only replacements parses fine and yields nothing:
        // those items were understood, just deliberately skipped.
        let replace_only = r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>cs.CL updates on arXiv.org</title>
    <item>
      <title>An Updated Paper</title>
      <link>https://arxiv.org/abs/2501.00003</link>
      <description>arXiv:2501.00003v2 Announce Type: replace
Abstract: An updated version of an older paper.</description>
      <guid isPermaLink="false">oai:arXiv.org:2501.00003v2</guid>
    </item>
  </channel>
</rss>"#;
        let papers = parse_rss_feed(replace_only).expect("replace-only feed is not a format error");
        assert!(papers.is_empty(), "replacements are skipped, not kept");

        // A feed with no items at all is simply a quiet day.
        let empty = r#"<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>cs.CL updates on arXiv.org</title>
  </channel>
</rss>"#;
        let papers = parse_rss_feed(empty).expect("empty feed is not an error");
        assert!(papers.is_empty());
    }

    #[test]
    fn build_rss_url_lists_categories_on_the_rss_host() {
        let url = build_rss_url(&["cs.CL".to_string(), "cs.LG".to_string()]);
        assert_eq!(url, "https://rss.arxiv.org/rss/cs.CL+cs.LG");
        check_url_allowed(&url).expect("RSS URL must be allowed");
    }

    #[tokio::test]
    async fn fetch_today_makes_one_request_and_parses_items() {
        let xml = std::fs::read_to_string(fixture_path("sample_rss_feed.xml"))
            .expect("fixture must exist");
        let fetcher = FakeFetcher::new(vec![("https://rss.arxiv.org/rss/".to_string(), xml)]);
        let limiter = RateLimiter::new(Duration::ZERO);

        let papers = fetch_today(
            &fetcher,
            &limiter,
            &["cs.CL".to_string(), "cs.LG".to_string()],
        )
        .await
        .expect("fetch should succeed");

        assert_eq!(
            papers.len(),
            2,
            "new + cross items survive, replace is dropped"
        );
        let requested = fetcher.requested.lock().await.clone();
        assert_eq!(requested.len(), 1, "exactly one request, no paging");
        assert_eq!(
            requested[0], "https://rss.arxiv.org/rss/cs.CL+cs.LG",
            "unexpected URL: {}",
            requested[0]
        );
    }

    #[tokio::test]
    async fn rss_url_is_rejected_when_host_not_allowed() {
        // The allowlist gate is enforced before any request: a feed URL on a
        // foreign host never reaches the fetcher.
        let err = check_url_allowed("https://evil.example.com/rss/cs.CL")
            .expect_err("foreign host must be rejected");
        assert!(
            matches!(err, ArxivError::HostNotAllowed(ref host) if host == "evil.example.com"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn build_query_url_targets_arxiv_window_and_categories() {
        let from = Utc.with_ymd_and_hms(2025, 1, 1, 9, 30, 0).unwrap();
        let to = Utc.with_ymd_and_hms(2025, 1, 3, 9, 30, 0).unwrap();
        let url = build_query_url(
            &["cs.CL".to_string(), "cs.LG".to_string()],
            from,
            to,
            200,
            200,
        );
        assert!(url.starts_with("https://export.arxiv.org/api/query?"));
        // arXiv's documented date-range syntax (GMT, to the minute) ANDed
        // with the category group, plus paging parameters.
        assert!(url.contains(
            "search_query=submittedDate:[202501010930+TO+202501030930]\
             +AND+(cat:cs.CL+OR+cat:cs.LG)&"
        ));
        assert!(url.contains("sortBy=submittedDate"));
        assert!(url.contains("sortOrder=descending"));
        assert!(url.contains("&start=200&max_results=200"));

        // The produced URL must itself pass the allowlist check.
        check_url_allowed(&url).expect("query URL must be allowed");
    }

    #[tokio::test]
    async fn fetch_by_query_stops_on_first_empty_page() {
        // Page 0 serves the two-entry fixture; every later page is empty.
        let page_one =
            std::fs::read_to_string(fixture_path("sample_feed.xml")).expect("fixture must exist");
        let fetcher = FakeFetcher::new(vec![
            ("&start=0&".to_string(), page_one),
            (String::new(), empty_feed_xml().to_string()),
        ]);
        let limiter = RateLimiter::new(Duration::from_millis(40));
        let now = Utc.with_ymd_and_hms(2025, 1, 3, 9, 30, 0).unwrap();
        let started = std::time::Instant::now();

        let papers = fetch_by_query(
            &fetcher,
            &limiter,
            &["cs.CL".to_string()],
            now,
            Duration::from_secs(48 * 3600),
        )
        .await
        .expect("fetch should succeed");
        let elapsed = started.elapsed();

        assert_eq!(papers.len(), 2, "both fixture papers are returned");
        let requested = fetcher.requested.lock().await.clone();
        assert_eq!(
            requested.len(),
            2,
            "paging must stop at the first empty page, requested: {requested:?}"
        );
        assert!(requested[0].contains("&start=0&"), "{}", requested[0]);
        assert!(
            requested[1].contains("&start=200&"),
            "second page starts one PAGE_SIZE further: {}",
            requested[1]
        );
        // The window is the 48h before `now`, in arXiv's GMT syntax.
        assert!(
            requested[0].contains("submittedDate:[202501010930+TO+202501030930]+AND+(cat:cs.CL)&"),
            "unexpected first query: {}",
            requested[0]
        );
        // Pages go through the rate limiter: the second request had to wait.
        assert!(
            elapsed >= Duration::from_millis(35),
            "pages must be spaced by the rate limiter, took {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn fetch_by_query_hard_caps_at_max_papers() {
        // Every page comes back full, so only the cap ends the run.
        let fetcher = FakeFetcher::new(vec![(String::new(), feed_with_n_entries(PAGE_SIZE))]);
        let limiter = RateLimiter::new(Duration::ZERO);
        let now = Utc.with_ymd_and_hms(2025, 1, 3, 9, 30, 0).unwrap();

        let papers = fetch_by_query(
            &fetcher,
            &limiter,
            &["cs.CL".to_string()],
            now,
            Duration::from_secs(48 * 3600),
        )
        .await
        .expect("fetch should succeed");

        assert_eq!(papers.len(), MAX_PAPERS, "result is capped");
        let requested = fetcher.requested.lock().await.clone();
        assert_eq!(
            requested.len(),
            MAX_PAPERS / PAGE_SIZE,
            "one request per full page, no more: {requested:?}"
        );
        assert!(
            !requested.iter().any(|u| u.contains("&start=2000&")),
            "must not page past the cap: {requested:?}"
        );
        assert!(requested[0].contains("&start=0&"));
        assert!(
            requested.last().is_some_and(|u| u.contains("&start=1800&")),
            "last page starts just below the cap: {requested:?}"
        );
    }

    #[tokio::test]
    async fn redirect_to_disallowed_host_or_plain_http_is_an_error() {
        // Hop to a host outside the allowlist.
        let fetcher = FakeFetcher::new(vec![]).with_redirects(vec![(
            "export.arxiv.org/api/query".to_string(),
            "https://evil.example.com/feed".to_string(),
        )]);
        let err = fetcher
            .fetch("https://export.arxiv.org/api/query?search_query=cat:cs.CL")
            .await
            .expect_err("redirect off the allowlist must fail loudly");
        assert!(
            matches!(err, ArxivError::Request { .. }),
            "should surface as a request error: {err}"
        );
        assert!(
            err.to_string().contains("redirect to disallowed host"),
            "unexpected error: {err}"
        );
        assert_eq!(
            fetcher.requested.lock().await.len(),
            1,
            "the disallowed hop is never fetched"
        );

        // Hop to an allowed host but over plain HTTP.
        let fetcher = FakeFetcher::new(vec![]).with_redirects(vec![(
            "export.arxiv.org/api/query".to_string(),
            "http://arxiv.org/api/query".to_string(),
        )]);
        let err = fetcher
            .fetch("https://export.arxiv.org/api/query?search_query=cat:cs.CL")
            .await
            .expect_err("plain HTTP redirect must be rejected");
        assert!(
            err.to_string().contains("redirect to disallowed host"),
            "non-HTTPS hop must be refused: {err}"
        );
    }

    #[tokio::test]
    async fn redirect_to_allowed_https_host_is_followed() {
        let fetcher = FakeFetcher::new(vec![(
            "arxiv.org/api/query".to_string(),
            "<feed/>".to_string(),
        )])
        .with_redirects(vec![(
            "export.arxiv.org/api/query".to_string(),
            "https://arxiv.org/api/query?moved=1".to_string(),
        )]);

        let body = fetcher
            .fetch("https://export.arxiv.org/api/query?search_query=cat:cs.CL")
            .await
            .expect("redirect to an allowed HTTPS host should be followed");

        assert_eq!(body, "<feed/>");
        let requested = fetcher.requested.lock().await.clone();
        assert_eq!(requested.len(), 2, "original request plus one hop");
        assert!(requested[1].contains("arxiv.org/api/query?moved=1"));
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
