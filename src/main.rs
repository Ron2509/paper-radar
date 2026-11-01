mod arxiv;
mod config;
mod llm;
mod secrets;

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
        Command::Digest { dry_run } => cmd_digest(dry_run).await,
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

async fn cmd_digest(dry_run: bool) -> anyhow::Result<()> {
    let _config = config::Config::load_default().context("loading config")?;
    let _soket_key = secrets::load_soket_key().context("loading Soket API key")?;
    if !dry_run {
        let _slack = secrets::SlackSecrets::load().context("loading Slack tokens")?;
    }
    println!("digest: not implemented yet");
    Ok(())
}

async fn cmd_run() -> anyhow::Result<()> {
    let _config = config::Config::load_default().context("loading config")?;
    let _soket_key = secrets::load_soket_key().context("loading Soket API key")?;
    let _slack = secrets::SlackSecrets::load().context("loading Slack tokens")?;
    println!("run: not implemented yet");
    Ok(())
}
