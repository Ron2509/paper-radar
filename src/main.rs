mod arxiv;
mod config;
mod digest;
mod filter;
mod llm;
mod ranking;
mod secrets;
mod slack;

use anyhow::Context;
use clap::{Parser, Subcommand};

/// paper-radar: daily arXiv digest bot for the team Slack.
#[derive(Parser)]
#[command(name = "paper-radar", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch today's new arXiv papers and print them.
    Fetch {
        /// Print to the terminal only (no side effects).
        #[arg(long)]
        dry_run: bool,
    },
    /// Run the full digest pipeline and post it to Slack.
    Digest {
        /// Print the digest to the terminal instead of posting to Slack.
        #[arg(long)]
        dry_run: bool,
        /// Read papers from a saved RSS/Atom feed file instead of arXiv.
        #[arg(long)]
        feed_file: Option<std::path::PathBuf>,
    },
    /// Long-running mode: Socket Mode listener plus the daily digest schedule.
    Run,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();

    match cli.command {
        // Fetching only reads the public arXiv feed, so both modes behave the
        // same for now; the flag exists for consistency with the other commands.
        Command::Fetch { dry_run: _ } => cmd_fetch().await,
        Command::Digest { dry_run, feed_file } => cmd_digest(dry_run, feed_file).await,
        Command::Run => cmd_run().await,
    }
}

async fn cmd_fetch() -> anyhow::Result<()> {
    let config = config::Config::load_default().context("loading config")?;
    let fetcher =
        arxiv::ReqwestFetcher::new(&config.arxiv.contact_email).context("building arXiv client")?;
    let limiter = arxiv::RateLimiter::new(arxiv::MIN_REQUEST_INTERVAL);
    let papers = arxiv::fetch_today(&fetcher, &limiter, &config.arxiv.categories).await?;

    println!("fetched {} new papers", papers.len());
    for paper in &papers {
        println!(
            "{} | {} | {}",
            paper.id,
            paper.title,
            paper.categories.join(", ")
        );
    }
    Ok(())
}

async fn cmd_digest(dry_run: bool, feed_file: Option<std::path::PathBuf>) -> anyhow::Result<()> {
    let config = config::Config::load_default().context("loading config")?;
    let soket_key = secrets::load_soket_key().context("loading Soket API key")?;
    let slack = if dry_run {
        None
    } else {
        Some(secrets::SlackSecrets::load().context("loading Slack tokens")?)
    };

    let papers = match &feed_file {
        Some(path) => {
            let body = std::fs::read_to_string(path)
                .with_context(|| format!("reading feed file {}", path.display()))?;
            arxiv::parse_rss_feed(&body).context("parsing saved feed file")?
        }
        None => {
            let fetcher = arxiv::ReqwestFetcher::new(&config.arxiv.contact_email)
                .context("building arXiv client")?;
            let limiter = arxiv::RateLimiter::new(arxiv::MIN_REQUEST_INTERVAL);
            arxiv::fetch_today(&fetcher, &limiter, &config.arxiv.categories).await?
        }
    };

    let fetched = papers.len();
    let kept = filter::keyword_filter(papers, &config.topics.keywords);
    let keywords_kept = kept.len();

    let llm = llm::SoketLlm::new(soket_key.expose(), &config.model.model).await?;
    let scored = ranking::score_papers(&llm, &kept, &config.topics.topics).await;
    let picks = digest::select_papers(scored, config.digest.min_score, config.digest.max_papers);

    println!(
        "fetched {} -> keywords {} -> scored >= {}: {}",
        fetched,
        keywords_kept,
        config.digest.min_score,
        picks.len()
    );

    if picks.is_empty() {
        println!("No strongly relevant papers today");
        if let Some(slack) = &slack {
            if config.digest.post_when_empty {
                let ts = slack::post_message(
                    slack.bot_token.expose(),
                    &config.slack.channel_id,
                    &slack::sanitize_for_slack("No strongly relevant papers today"),
                    None,
                )
                .await?;
                println!("posted to Slack, ts {ts}");
            }
        }
        return Ok(());
    }

    let text = digest::build_digest(&llm, &picks).await?;
    if dry_run {
        print!("{text}");
    } else {
        let slack = slack.expect("slack secrets loaded when not dry-run");
        let ts = slack::post_message(
            slack.bot_token.expose(),
            &config.slack.channel_id,
            &slack::sanitize_for_slack(&text),
            None,
        )
        .await?;
        println!("posted to Slack, ts {ts}");
    }
    Ok(())
}

async fn cmd_run() -> anyhow::Result<()> {
    let _config = config::Config::load_default().context("loading config")?;
    let _soket_key = secrets::load_soket_key().context("loading Soket API key")?;
    let _slack = secrets::SlackSecrets::load().context("loading Slack tokens")?;
    println!("run: not implemented yet");
    Ok(())
}
