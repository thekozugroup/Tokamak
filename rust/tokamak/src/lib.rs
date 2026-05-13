//! Tokamak — terse-reasoning trace processor and curator.
//!
//! Single-binary pipeline. Each row's reasoning is processed by an agent
//! (compress / invert / both / rules / noop) with the full PROBLEM + ANSWER
//! as reference rails. Optional QAQC validator grades each span on 0..1 and
//! the score lands in the row's `signal` column.

pub mod anonymize;
pub mod card;
pub mod classify;
pub mod dedup;
pub mod export;
pub mod extract;
pub mod llm;
pub mod pipeline;
pub mod processor;
pub mod prompts;
pub mod quality;
pub mod terse;
pub mod validator;

pub use pipeline::{run, RunOptions, RunStats};
