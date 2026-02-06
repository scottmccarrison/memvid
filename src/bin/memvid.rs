use memvid_core::{Memvid, PutOptions, SearchRequest};
#[cfg(feature = "vec")]
use memvid_core::{DoctorOptions, LocalTextEmbedder, TextEmbedConfig};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

/// Global override set by --memory flag before command dispatch
static MEMORY_PATH_OVERRIDE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

fn get_memory_path() -> PathBuf {
    // 1. --memory CLI flag (highest priority)
    if let Some(p) = MEMORY_PATH_OVERRIDE.get() {
        return p.clone();
    }
    // 2. MEMVID_MEMORY env var
    if let Ok(p) = env::var("MEMVID_MEMORY") {
        let expanded = if p.starts_with('~') {
            let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
            p.replacen('~', &home, 1)
        } else {
            p
        };
        return PathBuf::from(expanded);
    }
    // 3. Default fallback
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".memvid").join("claude.mv2")
}

fn ensure_memory_dir() -> io::Result<()> {
    let path = get_memory_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

#[cfg(feature = "vec")]
fn get_embedder() -> Result<LocalTextEmbedder, Box<dyn std::error::Error>> {
    let config = TextEmbedConfig::default();
    Ok(LocalTextEmbedder::new(config)?)
}

fn cmd_save(title: Option<&str>, tags: Vec<(&str, &str)>, content: &str) -> Result<(), Box<dyn std::error::Error>> {
    ensure_memory_dir()?;
    let path = get_memory_path();

    let mut mem = if path.exists() {
        Memvid::open(&path)?
    } else {
        Memvid::create(&path)?
    };

    let mut opts = PutOptions::builder();

    if let Some(t) = title {
        opts = opts.title(t);
    }

    for (key, value) in tags {
        opts = opts.tag(key, value);
    }

    // Generate embedding if vec feature is enabled
    #[cfg(feature = "vec")]
    let seq = {
        match get_embedder() {
            Ok(embedder) => {
                match embedder.encode_text(content) {
                    Ok(embedding) => {
                        mem.put_with_embedding_and_options(content.as_bytes(), embedding, opts.build())?
                    }
                    Err(e) => {
                        eprintln!("Warning: Could not generate embedding ({}), saving without", e);
                        mem.put_bytes_with_options(content.as_bytes(), opts.build())?
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Could not load embedder ({}), saving without", e);
                mem.put_bytes_with_options(content.as_bytes(), opts.build())?
            }
        }
    };

    #[cfg(not(feature = "vec"))]
    let seq = mem.put_bytes_with_options(content.as_bytes(), opts.build())?;

    mem.commit()?;

    println!("Saved to memory (frame {})", seq);
    Ok(())
}

fn cmd_search(query: &str, top_k: usize) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_memory_path();

    if !path.exists() {
        println!("No memory file found. Save something first with 'memvid save'");
        return Ok(());
    }

    let mut mem = Memvid::open(&path)?;

    // Run lexical search (always available with lex feature)
    let request = SearchRequest {
        query: query.to_string(),
        top_k: top_k * 2, // Get more results for hybrid merging
        snippet_chars: 300,
        uri: None,
        scope: None,
        cursor: None,
        #[cfg(feature = "temporal_track")]
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
        no_sketch: false,
    };

    let lex_response = mem.search(request)?;

    // Try hybrid search with vec if available
    #[cfg(feature = "vec")]
    let final_hits = {
        let stats = mem.stats()?;
        if stats.has_vec_index {
            match get_embedder() {
                Ok(embedder) => {
                    match embedder.encode_text(query) {
                        Ok(query_embedding) => {
                            match mem.vec_search_with_embedding(query, &query_embedding, top_k * 2, 300, None) {
                                Ok(vec_response) => {
                                    // Hybrid merge using Reciprocal Rank Fusion
                                    merge_results_rrf(&lex_response.hits, &vec_response.hits, top_k)
                                }
                                Err(_) => {
                                    // Fall back to lex only
                                    lex_response.hits.into_iter().take(top_k).collect()
                                }
                            }
                        }
                        Err(_) => lex_response.hits.into_iter().take(top_k).collect(),
                    }
                }
                Err(_) => lex_response.hits.into_iter().take(top_k).collect(),
            }
        } else {
            lex_response.hits.into_iter().take(top_k).collect()
        }
    };

    #[cfg(not(feature = "vec"))]
    let final_hits: Vec<_> = lex_response.hits.into_iter().take(top_k).collect();

    if final_hits.is_empty() {
        println!("No results found for: {}", query);
        return Ok(());
    }

    println!("Found {} results ({} ms):\n", final_hits.len(), lex_response.elapsed_ms);

    for hit in final_hits {
        let title = hit.title.as_deref().unwrap_or("Untitled");
        let score = hit.score.unwrap_or(0.0);
        println!("--- [{}] {} (score: {:.3}) ---", hit.frame_id, title, score);
        println!("{}\n", hit.text.trim());
    }

    Ok(())
}

/// Merge search results using Reciprocal Rank Fusion (RRF)
/// This gives good results even when scores from different systems aren't comparable
#[cfg(feature = "vec")]
fn merge_results_rrf(
    lex_hits: &[memvid_core::SearchHit],
    vec_hits: &[memvid_core::SearchHit],
    top_k: usize,
) -> Vec<memvid_core::SearchHit> {
    use std::collections::HashSet;

    const K: f32 = 60.0; // RRF constant - standard value

    let mut scores: HashMap<u64, f32> = HashMap::new();
    let mut hits_by_id: HashMap<u64, memvid_core::SearchHit> = HashMap::new();

    // Add lexical results with RRF scores
    for (rank, hit) in lex_hits.iter().enumerate() {
        let rrf_score = 1.0 / (K + (rank + 1) as f32);
        *scores.entry(hit.frame_id).or_insert(0.0) += rrf_score;
        hits_by_id.entry(hit.frame_id).or_insert_with(|| hit.clone());
    }

    // Add vector results with RRF scores
    for (rank, hit) in vec_hits.iter().enumerate() {
        let rrf_score = 1.0 / (K + (rank + 1) as f32);
        *scores.entry(hit.frame_id).or_insert(0.0) += rrf_score;
        hits_by_id.entry(hit.frame_id).or_insert_with(|| hit.clone());
    }

    // Sort by combined RRF score
    let mut scored: Vec<_> = scores.into_iter().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Return top_k hits with updated scores
    let mut seen = HashSet::new();
    scored
        .into_iter()
        .filter_map(|(frame_id, rrf_score)| {
            if seen.insert(frame_id) {
                hits_by_id.remove(&frame_id).map(|mut hit| {
                    hit.score = Some(rrf_score);
                    hit
                })
            } else {
                None
            }
        })
        .take(top_k)
        .collect()
}

fn cmd_stats() -> Result<(), Box<dyn std::error::Error>> {
    let path = get_memory_path();

    if !path.exists() {
        println!("No memory file found at {:?}", path);
        return Ok(());
    }

    let mem = Memvid::open(&path)?;
    let stats = mem.stats()?;

    println!("Memory Statistics:");
    println!("  Location: {:?}", path);
    println!("  Frames: {}", stats.frame_count);
    println!("  Active frames: {}", stats.active_frame_count);
    println!("  Size: {} bytes", stats.size_bytes);
    println!("  Has lex index: {}", stats.has_lex_index);
    #[cfg(feature = "vec")]
    println!("  Has vec index: {}", stats.has_vec_index);
    println!("  Compression: {:.1}%", stats.compression_ratio_percent);

    Ok(())
}

fn cmd_list_recent(count: usize) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_memory_path();

    if !path.exists() {
        println!("No memory file found at {:?}", path);
        return Ok(());
    }

    let mem = Memvid::open(&path)?;
    let stats = mem.stats()?;
    let total = stats.frame_count as usize;

    println!("Total frames: {}", total);
    println!("Listing last {} frames:", count);

    let start = if total > count { total - count } else { 0 };
    for i in start..total {
        match mem.frame_by_id(i as u64) {
            Ok(frame) => {
                let has_search = if frame.search_text.is_some() { "✓" } else { "✗" };
                let has_mime = if frame.metadata.as_ref().and_then(|m| m.mime.as_ref()).is_some() { "✓" } else { "✗" };
                let title_preview = frame.title.as_deref().unwrap_or("(no title)");
                let title_short = if title_preview.len() > 40 { &title_preview[..40] } else { title_preview };
                println!("  [{}] search:{} mime:{} {:?}", i, has_search, has_mime, title_short);
            }
            Err(e) => {
                println!("  [{}] ERROR: {}", i, e);
            }
        }
    }

    Ok(())
}

fn cmd_test_save_search() -> Result<(), Box<dyn std::error::Error>> {
    ensure_memory_dir()?;
    let path = get_memory_path();

    println!("Opening/creating memvid at {:?}", path);

    let mut mem = if path.exists() {
        println!("Opening existing file...");
        Memvid::open(&path)?
    } else {
        println!("Creating new file...");
        Memvid::create(&path)?
    };

    let stats = mem.stats()?;
    println!("Stats after open: frames={}, has_lex={}", stats.frame_count, stats.has_lex_index);

    // Save unique test content
    let unique = format!("TESTUNIQ_{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs());

    println!("Saving unique content: {}", unique);

    let opts = PutOptions::builder()
        .title("Save-Search Test");

    let seq = mem.put_bytes_with_options(unique.as_bytes(), opts.build())?;
    println!("Saved as frame sequence {}", seq);

    mem.commit()?;
    println!("Committed.");

    // Now search for it in the same session
    // Try searching for "testuniq" which is simpler
    println!("\nSearching for 'testuniq' in same session...");

    let request = SearchRequest {
        query: "testuniq".to_string(),
        top_k: 5,
        snippet_chars: 300,
        uri: None,
        scope: None,
        cursor: None,
        #[cfg(feature = "temporal_track")]
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
        no_sketch: false,
    };

    let response = mem.search(request)?;

    if response.hits.is_empty() {
        println!("❌ No results found in SAME session!");
    } else {
        println!("✓ Found {} results in same session:", response.total_hits);
        for hit in &response.hits {
            println!("  [{}] {}", hit.frame_id, hit.text.chars().take(50).collect::<String>());
        }
    }

    drop(mem);

    // Now reopen and search again
    println!("\nReopening file and searching again...");
    let mut mem2 = Memvid::open(&path)?;

    let request2 = SearchRequest {
        query: "testuniq".to_string(),
        top_k: 5,
        snippet_chars: 300,
        uri: None,
        scope: None,
        cursor: None,
        #[cfg(feature = "temporal_track")]
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
        no_sketch: false,
    };

    let response2 = mem2.search(request2)?;

    if response2.hits.is_empty() {
        println!("❌ No results found after REOPEN!");
    } else {
        println!("✓ Found {} results after reopen:", response2.total_hits);
        for hit in &response2.hits {
            println!("  [{}] {}", hit.frame_id, hit.text.chars().take(50).collect::<String>());
        }
    }

    Ok(())
}

fn cmd_inspect(frame_id: u64) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_memory_path();

    if !path.exists() {
        println!("No memory file found at {:?}", path);
        return Ok(());
    }

    let mem = Memvid::open(&path)?;

    match mem.frame_by_id(frame_id) {
        Ok(frame) => {
            println!("Frame {}:", frame_id);
            println!("  Status: {:?}", frame.status);
            println!("  Timestamp: {}", frame.timestamp);
            println!("  Title: {:?}", frame.title);
            println!("  URI: {:?}", frame.uri);
            println!("  Search text present: {}", frame.search_text.is_some());
            if let Some(ref text) = frame.search_text {
                let preview = if text.len() > 100 { &text[..100] } else { text };
                println!("  Search text preview: {:?}", preview);
            }
            println!("  Tags: {:?}", frame.tags);
            println!("  Labels: {:?}", frame.labels);
            println!("  Role: {:?}", frame.role);
            if let Some(ref meta) = frame.metadata {
                println!("  MIME: {:?}", meta.mime);
            } else {
                println!("  Metadata: None");
            }
            println!("  Payload length: {}", frame.payload_length);
            println!("  Canonical encoding: {:?}", frame.canonical_encoding);
        }
        Err(e) => {
            println!("Error getting frame {}: {}", frame_id, e);
        }
    }

    Ok(())
}

#[cfg(feature = "vec")]
fn cmd_doctor(rebuild_vec: bool, rebuild_lex: bool) -> Result<(), Box<dyn std::error::Error>> {
    let path = get_memory_path();

    if !path.exists() {
        println!("No memory file found at {:?}", path);
        return Ok(());
    }

    println!("Running doctor on {:?}...", path);

    let options = DoctorOptions {
        rebuild_vec_index: rebuild_vec,
        rebuild_lex_index: rebuild_lex,
        rebuild_time_index: false,
        dry_run: false,
        quiet: false,
        vacuum: false,
    };

    let report = Memvid::doctor(&path, options)?;

    println!("Doctor completed: {:?}", report.status);
    if let Some(v) = &report.verification {
        println!("  Verified: {:?}", v.overall_status);
    }

    Ok(())
}

#[cfg(feature = "vec")]
fn cmd_embed_all() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    let path = get_memory_path();

    if !path.exists() {
        println!("No memory file found at {:?}", path);
        return Ok(());
    }

    println!("Loading embedding model...");
    let embedder = get_embedder()?;

    println!("Opening memory file...");
    let mut mem = Memvid::open(&path)?;
    let stats = mem.stats()?;
    let total_frames = stats.frame_count;

    println!("Scanning {} frames for missing embeddings...", total_frames);

    // Collect frames that need embeddings
    let mut frames_to_embed: Vec<(u64, String)> = Vec::new();

    for frame_id in 0..total_frames {
        // Check if frame already has embedding
        if let Ok(Some(_)) = mem.frame_embedding(frame_id) {
            continue; // Already has embedding
        }

        // Get frame text
        match mem.frame_text_by_id(frame_id) {
            Ok(text) if !text.trim().is_empty() => {
                frames_to_embed.push((frame_id, text));
            }
            _ => continue, // Skip empty or errored frames
        }
    }

    let need_embedding = frames_to_embed.len();
    if need_embedding == 0 {
        println!("All frames already have embeddings!");
        return Ok(());
    }

    println!("Generating embeddings for {} frames...", need_embedding);
    let start = Instant::now();

    // Generate embeddings in batches
    let mut embeddings: Vec<(u64, Vec<f32>)> = Vec::with_capacity(need_embedding);
    let batch_size = 50;

    for (i, (frame_id, text)) in frames_to_embed.iter().enumerate() {
        match embedder.encode_text(text) {
            Ok(embedding) => {
                embeddings.push((*frame_id, embedding));
            }
            Err(e) => {
                eprintln!("  Warning: Failed to embed frame {}: {}", frame_id, e);
            }
        }

        // Progress update every batch_size frames
        if (i + 1) % batch_size == 0 || i + 1 == need_embedding {
            let elapsed = start.elapsed().as_secs_f32();
            let rate = (i + 1) as f32 / elapsed;
            let remaining = (need_embedding - i - 1) as f32 / rate;
            print!("\r  Progress: {}/{} ({:.0}/sec, ~{:.0}s remaining)    ",
                   i + 1, need_embedding, rate, remaining);
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!(); // Newline after progress

    if embeddings.is_empty() {
        println!("No embeddings generated.");
        return Ok(());
    }

    println!("Adding {} embeddings to index...", embeddings.len());
    let added = mem.add_embeddings(embeddings)?;

    println!("Committing changes...");
    mem.commit()?;

    let elapsed = start.elapsed();
    println!("Done! Added {} embeddings in {:.1}s", added, elapsed.as_secs_f32());

    Ok(())
}

fn print_usage() {
    let path = get_memory_path();
    eprintln!("Usage: memvid [--memory <path>] <command> [args]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  save [--title <title>] [--tag key=value]... <content>");
    eprintln!("  save --stdin [--title <title>] [--tag key=value]...");
    eprintln!("  search <query> [--top <n>]");
    eprintln!("  stats");
    eprintln!("  list [count]                             List recent frames");
    eprintln!("  inspect <frame_id>                       Show frame details");
    #[cfg(feature = "vec")]
    eprintln!("  embed-all                                Generate embeddings for all frames");
    #[cfg(feature = "vec")]
    eprintln!("  doctor [--rebuild-lex] [--rebuild-vec]   Rebuild indexes");
    eprintln!();
    eprintln!("Memory path (in priority order):");
    eprintln!("  1. --memory <path>      CLI flag");
    eprintln!("  2. $MEMVID_MEMORY       Environment variable");
    eprintln!("  3. ~/.memvid/claude.mv2 Default");
    eprintln!();
    eprintln!("Active: {}", path.display());
    #[cfg(feature = "vec")]
    eprintln!("Hybrid search (lex + semantic) enabled.");
}

fn main() {
    let raw_args: Vec<String> = env::args().collect();

    // Parse global --memory flag before command dispatch
    let args: Vec<String> = {
        let mut filtered = vec![raw_args[0].clone()];
        let mut i = 1;
        while i < raw_args.len() {
            if raw_args[i] == "--memory" || raw_args[i] == "-m" {
                if i + 1 < raw_args.len() {
                    let p = &raw_args[i + 1];
                    let expanded = if p.starts_with('~') {
                        let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
                        p.replacen('~', &home, 1)
                    } else {
                        p.clone()
                    };
                    let _ = MEMORY_PATH_OVERRIDE.set(PathBuf::from(expanded));
                    i += 2;
                    continue;
                } else {
                    eprintln!("Missing path for --memory flag");
                    std::process::exit(1);
                }
            }
            filtered.push(raw_args[i].clone());
            i += 1;
        }
        filtered
    };

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "save" => {
            let mut title: Option<&str> = None;
            let mut tags: Vec<(&str, &str)> = Vec::new();
            let mut content = String::new();
            let mut use_stdin = false;
            let mut i = 2;

            while i < args.len() {
                match args[i].as_str() {
                    "--title" | "-t" => {
                        if i + 1 < args.len() {
                            title = Some(&args[i + 1]);
                            i += 2;
                        } else {
                            eprintln!("Missing title value");
                            std::process::exit(1);
                        }
                    }
                    "--tag" => {
                        if i + 1 < args.len() {
                            if let Some((k, v)) = args[i + 1].split_once('=') {
                                tags.push((k, v));
                            }
                            i += 2;
                        } else {
                            eprintln!("Missing tag value");
                            std::process::exit(1);
                        }
                    }
                    "--stdin" => {
                        use_stdin = true;
                        i += 1;
                    }
                    _ => {
                        content = args[i..].join(" ");
                        break;
                    }
                }
            }

            if use_stdin {
                io::stdin().read_to_string(&mut content).expect("Failed to read stdin");
            }

            if content.trim().is_empty() {
                eprintln!("No content provided");
                std::process::exit(1);
            }

            // Convert owned strings to references for the function call
            let tags_refs: Vec<(&str, &str)> = tags.iter().map(|(k, v)| (*k, *v)).collect();
            cmd_save(title, tags_refs, &content)
        }
        "search" | "find" => {
            let mut query = String::new();
            let mut top_k = 5;
            let mut i = 2;

            while i < args.len() {
                match args[i].as_str() {
                    "--top" | "-n" => {
                        if i + 1 < args.len() {
                            top_k = args[i + 1].parse().unwrap_or(5);
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    _ => {
                        if query.is_empty() {
                            query = args[i..].join(" ");
                        }
                        break;
                    }
                }
            }

            if query.is_empty() {
                eprintln!("No search query provided");
                std::process::exit(1);
            }

            cmd_search(&query, top_k)
        }
        "stats" => cmd_stats(),
        "inspect" => {
            if args.len() < 3 {
                eprintln!("Usage: memvid inspect <frame_id>");
                std::process::exit(1);
            }
            let frame_id: u64 = args[2].parse().unwrap_or_else(|_| {
                eprintln!("Invalid frame ID: {}", args[2]);
                std::process::exit(1);
            });
            cmd_inspect(frame_id)
        }
        "list" | "ls" => {
            let count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
            cmd_list_recent(count)
        }
        "test-save-search" => cmd_test_save_search(),
        #[cfg(feature = "vec")]
        "doctor" => {
            let rebuild_vec = args.iter().any(|a| a == "--rebuild-vec");
            let rebuild_lex = args.iter().any(|a| a == "--rebuild-lex");
            cmd_doctor(rebuild_vec, rebuild_lex)
        }
        #[cfg(feature = "vec")]
        "embed-all" => cmd_embed_all(),
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
