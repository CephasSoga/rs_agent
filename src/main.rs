#![allow(unused)]


mod memory;
mod tools;
mod config;
mod ops;
mod looging;
mod utils;

use std::time::Instant;
use tools::deep_search::ExhautNodes;
use tracing::info;

use crate::tools::searxng::{HtmlParser, RequestHandler, MetaSearchEngine};
use crate::tools::deep_search::DeepSearchEngine;
use crate::looging::setup_logger;




#[tokio::main]
async fn main() {
    setup_logger("info");
    //let engine = MetaSearchEngine::new("http://localhost:8080/search");
    //let r = engine.search("how is the war in ukraine evolving", Some("json"), "p").await;

    time!({let mut engine = DeepSearchEngine::new("http://localhost:8080/search");
    //let mut roots = engine.fetch_roots("how is the war in ukraine evolving").await.unwrap();
    //let _ = engine.pretty_save_to_file("tree.json", &tree.clone());
    let tree = engine.run("how is the war in ukraine evolving", 3).await;
    let tree = tree.unwrap();
    let _ = engine.pretty_save_to_file("tree.json", &tree.clone());

    info!("Explored links: {:#?}", &engine.explored);})
}

use scraper::{Html, Selector};

fn extract_text_and_links(html: &str) {
    let document = Html::parse_document(html);

    // Select all <p> tags
    let paragraph_selector = Selector::parse("p").unwrap();
    for element in document.select(&paragraph_selector) {
        println!("Text: {}", element.text().collect::<Vec<_>>().join(" "));
    }

    // Select all <a> tags
    let link_selector = Selector::parse("a").unwrap();
    for element in document.select(&link_selector) {
        let link_text = element.text().collect::<Vec<_>>().join(" ");
        let href = element.value().attr("href").unwrap_or("#");
        println!("Link: {} -> {}", link_text, href);
    }
}

fn main_() {
    let html = r#"
        <html>
            <body>
                <p>This is a simple paragraph with two useful links.</p>
                <a href="https://www.rust-lang.org">Learn Rust</a>
                <a href="https://www.python.org">Learn Python</a>
            </body>
        </html>
    "#;

    extract_text_and_links(html);
}

pub struct Extractor;

impl Extractor {
    pub async fn extract(&self, html: &str, selector_str: &str) -> Vec<String> {
        let document = Html::parse_document(html);
        let selector = Selector::parse(selector_str).unwrap();
        let elements = document.select(&selector);

        match selector_str {
            "a" => elements
                .map(|el| el.value().attr("href").unwrap_or("#").to_string())
                .collect(),
            _ => elements
                .map(|el| el.text().collect::<Vec<_>>().join(" "))
                .collect(),
        }
    }
}

#[tokio::main]
async fn main_2() {
    let html = r#"
        <html>
            <body>
                <p>This is a simple paragraph.</p>
                <a href="https://www.rust-lang.org">Learn Rust</a>
                <a href="https://www.python.org">Learn Python</a>
            </body>
        </html>
    "#;

    let extractor = Extractor;

    let paragraphs = extractor.extract(html, "p").await;
    let links = extractor.extract(html, "a").await;

    println!("Paragraphs: {:?}", paragraphs);
    println!("Links: {:?}", links);
}
