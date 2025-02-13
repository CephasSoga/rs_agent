#![allow(dead_code)]

use std::fmt::Debug;
use std::os::raw;
use std::sync::Arc;
use std::cmp::Ordering;
use std::collections::HashSet;

use dashmap::DashSet;
use scraper::html;
use serde_json::Value;
use serde::{de, Deserialize, Serialize};
use regex::Regex;
use tokio::{spawn, sync::Mutex, task::JoinHandle};
use tracing::{debug, error, info, warn};
use tracing_subscriber::field::debug;

use super::searxng::{HtmlParser, RequestHandler, MetaSearchEngine};
use super::errors::EngineError;
use crate::ops::embedding::Word2VecFromFile;
use super::debug::HTML;


const EMBEDDING_PAD_LEN: usize = 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub depth_index: u32,
    pub url: String,
    pub relevance: f64,
    pub explored: bool,
    pub extracted: Option<HtmlWrap>
}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.url == other.url
    }
}
impl Eq for Node {}
impl Node {
    fn update(&mut self, score: f64) {
        self.relevance = score;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlWrap {
    pub title: Vec<String>,
    pub h1: Vec<String>,
    pub h2: Vec<String>,
    pub h3: Vec<String>,
    pub h4: Vec<String>,
    pub h5: Vec<String>,
    pub h6: Vec<String>,
    pub text: Vec<String>,
    pub links: Vec<String>,
}
impl Default for HtmlWrap {
    fn default() -> Self {
        Self {
            title: Vec::new(),
            h1: Vec::new(),
            h2: Vec::new(),
            h3: Vec::new(),
            h4: Vec::new(),
            h5: Vec::new(),
            h6: Vec::new(),
            text: Vec::new(),
            links: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode<T> {
    pub node: T,
    pub children: Vec<TreeNode<T>>
}
impl <T> TreeNode<T> {
    pub fn new(node: T) -> Self {
        Self {
            node,
            children: Vec::new()
        }
    }

    pub fn add_child(&mut self, child: TreeNode<T>) {
        self.children.push(child)
    }

    pub fn drop_child(&mut self, index: usize) {
        let _ = self.children.remove(index);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthLevel {
    pub level: u32,
    pub links: Vec<TreeNode<Node>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchTree {
    pub levels: Vec<DepthLevel>,
    pub query: String,
    pub depth: u32,
}

pub trait HtmlOps {
    fn filter_valid_links(potential_links: &[String]) -> Vec<String>;
}

pub trait Word2VecInitializer {
    fn initialize(&self, model_path: Option<&str>) -> Word2VecFromFile;
}

pub trait Word2VecEmbedding {
    type R; // the embedding result
    fn embed(&self, text: &str, len: usize) -> Self::R;
    fn pad(&self, text: &str, tokens_count: usize) -> String;
    fn cosine_similarity(a: &Self::R, b: &Self::R) -> f64;
}

pub trait ExhautNodes {
    type E; // The error type
    type R; // The embedding result
    type ArcEx; // Keeps trcak of explored links
    async fn fetch_roots(&self, query: &str) -> Result<Vec<TreeNode<Node>>, Self::E>;
    async fn fetch_children(&self, q_embed: &Self::R, parent: &mut TreeNode<Node>, depth: u32) -> Result<Vec<TreeNode<Node>>, Self::E>;
    async fn wrap_response(&self, q_emb: &Self::R, html: &str) -> HtmlWrap;
    async fn explore(&mut self, query: &str, nodes: Vec<TreeNode<Node>>, depth: u32) -> Result<SearchTree, EngineError>;
    async fn explore_node(&self, q_emb: &Option<Vec<f32>>, tree: TreeNode<Node>, depth: u32, result: Arc<Mutex<SearchTree>>, explored: Self::ArcEx);
    async fn score_link(&self, q_emb: &Self::R, link: &str) -> Result<f64, Self::E>;
}


pub struct DeepSearchEngine {
    engine: MetaSearchEngine,
    embedding_model: Option<Arc<Word2VecFromFile>>,
    pub explored: Arc<DashSet<String>>
}

impl DeepSearchEngine {
    pub fn new(host: &str) -> Self {
        Self {
            engine: MetaSearchEngine::new(host),
            embedding_model: None,
            explored: Arc::new(DashSet::new())
        
        }
    }
    fn clone(&self) -> Self {
        Self {
            engine: self.engine.clone(),
            embedding_model: self.embedding_model.clone(),
            explored: self.explored.clone()
        }
    }
    pub fn pretty_save_to_file(&self, filename: &str, tree: &SearchTree) -> Result<(), std::io::Error> {
        let mut file = std::fs::File::create(filename)?;
        let formatter = serde_json::ser::PrettyFormatter::with_indent(b"  "); // 2 spaces
        let mut serializer = serde_json::Serializer::with_formatter(&mut file, formatter);
        tree.serialize(&mut serializer)?;
        info!("Search tree is saved to file: {}.", filename);
        Ok(())
    }
    
    pub async fn run(&mut self, query: &str, depth: u32) -> Result<SearchTree, EngineError>  {
        info!("Intializing the embedding model...");
        if self.embedding_model.is_none() {
            self.embedding_model = Some(Arc::new(self.initialize(None)));
        }
        info!("Running deep search engine...");
        let root_nodes = self.fetch_roots(query).await?;
        let r = self.explore(query, root_nodes, depth).await;
        info!("Deep search engine finished.");
        r
    }
}

impl HtmlOps for DeepSearchEngine {
    fn filter_valid_links(potential_links: &[String]) -> Vec<String> {
        debug!("Filtering found links...");
        let url_regex = Regex::new(r#"(?i)\b(?:https?://|www\.)\S+\b"#).unwrap();
        
        potential_links
            .iter()
            .filter_map(|link| {
                url_regex
                    .find(link)
                    .map(|m| m.as_str().to_string())
            })
            .inspect(|link| debug!("Valid link found: {:?}", link))
            .collect()
    }    
}

impl Word2VecInitializer for DeepSearchEngine {
    fn initialize(&self, model_path: Option<&str>) -> Word2VecFromFile {
        Word2VecFromFile::new(model_path.unwrap_or("models/multi_thread_model.json")).unwrap()
    }
}

impl Word2VecEmbedding for DeepSearchEngine {
    type R = Option<Vec<f32>>;

    fn embed(&self, text: &str, len: usize) -> Self::R {
        self.embedding_model.as_ref().unwrap().get_embedding_for_text(&self.pad(&text, len))
    }

    fn pad(&self, text: &str, tokens_count: usize) -> String {
        debug!("Padding text: {}", text);
        let words: Vec<&str> = text.split_whitespace().collect();
        let margin = tokens_count as isize - words.len() as isize;
        if margin <= 0 {
            return text.split_whitespace().collect::<Vec<&str>>()[..tokens_count].join(" ")
        } 
        let padding = vec!["&c"; margin as usize]; // Because `&c` is present in the model's corpus.
        let padded_text = words.iter().chain(padding.iter()).map(|&w| w.to_string()).collect::<Vec<_>>().join(" ");
        padded_text
    }

    fn cosine_similarity(a: &Self::R, b: &Self::R) -> f64 {
        if let Some(a) = a {
            if let Some(b) = b {
                if a.len() != b.len() {
                    error!("Vectors must be of the same length! Retruned 0.");
                    return 0_f64;
                }
            
                let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            
                (dot_product / (norm_a * norm_b)) as f64
            } else {
                error!("Second vector is empty! Retruned 0.");
                0_f64
            }
        } else {
            error!("First vector is empty! Retruned 0.");
            0_f64
        }
    }
}

impl ExhautNodes for DeepSearchEngine {
    type E = EngineError;
    type R = Option<Vec<f32>>;
    type ArcEx = Arc<DashSet<String>>;
    async fn fetch_roots(&self, query: &str) -> Result<Vec<TreeNode<Node>>, Self::E> {
        info!("Fetching roots for query: {}.", query);
        let mut roots: Vec<TreeNode<Node>>= Vec::new();

        let urls = self.engine.collect_urls(query, Some("json")).await?;
        roots = urls.into_iter()
            .filter_map(|url| url.map(|u| TreeNode::new(Node {
                depth_index: 0,
                url: u,
                relevance: 1.0,
                explored: false,
                extracted: None,
            })))
            .collect::<Vec<_>>();

        debug!("Fetched roots: {:?}", roots);
        Ok(roots)
    }

    async fn fetch_children(&self, q_emb: &Option<Vec<f32>>, parent: &mut TreeNode<Node>, depth: u32) -> Result<Vec<TreeNode<Node>>, Self::E> {
        if depth == 0 {
            warn!("Depth is 0, not fetching children.");
            return Ok(Vec::new());
        }

        debug!("Fetching children for node. | Url: {:?}", parent.node.url);

        match self.engine.request(&parent.node.url, Some("html")).await {
            Ok(response) => {
                debug!("Fetched HTML for URL: <{}>.", &parent.node.url);
                
                if let Ok(text) = response.text().await {
                    
                    let html_wrap = self.wrap_response(q_emb,&text).await;
                    debug!("Wrapped HTML for URL: <{:#?}>.", &html_wrap);
                    
                    parent.node.extracted = Some(html_wrap.clone());
                    parent.node.explored = true;
                    let score = self.score_link(q_emb, &parent.node.url).await?;
                    parent.node.update(score);
                    self.explored.insert(parent.node.url.clone());

                    let child_nodes: Vec<TreeNode<Node>> = html_wrap.links.iter().map(|url| {
                        TreeNode::new(Node {
                            depth_index: parent.node.depth_index + 1,
                            url: url.clone(),
                            relevance: 1.0,
                            explored: false,
                            extracted: None,
                        })
                    }).collect();

                    parent.children.extend(child_nodes.clone()); // Add children to parent
                    return Ok(child_nodes);
                }
            }
            Err(err) => {
                error!("Failed to fetch children for URL: <{}>.", &parent.node.url);
                return Err(err);
            }  
        }
        // Return an empty vector if no children were fetched
        Ok(Vec::new())
    }
    
    async fn wrap_response(&self, q_emb: &Option<Vec<f32>>, html: &str) -> HtmlWrap {
        debug!("Wrapping HTML: <{}>.", html);
        let mut html_wrap = HtmlWrap::default();
        let selectors = vec![
            ("h1", &mut html_wrap.h1),
            ("h2", &mut html_wrap.h2),
            ("h3", &mut html_wrap.h3),
            ("h4", &mut html_wrap.h4),
            ("h5", &mut html_wrap.h5),
            ("h6", &mut html_wrap.h6),
            ("a", &mut html_wrap.links),
            ("p", &mut html_wrap.text),
            ("title", &mut html_wrap.title),
        ];
        for (field, target) in selectors {
            *target = self.engine.extract(html, field).await;
        }

        // filter valid links and keep unique ones.
        html_wrap.links = Self::filter_valid_links(&html_wrap.links)
            .iter()
            .cloned()
            .collect::<HashSet<String>>()
            .into_iter()
            .collect::<Vec<String>>();

        // Collect scores first
        let mut scored_links = Vec::new();
        for link in html_wrap.links.iter() {
            if let Ok(score) = self.score_link(q_emb, link).await {
                scored_links.push((score, link.clone()));
            }
        }
        // Sort by score
        scored_links.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        // Replace links with sorted ones
        html_wrap.links = scored_links.into_iter().map(|(_, link)| link).take(5).collect();
        
        html_wrap
    }

    async fn explore(&mut self, query: &str,  nodes: Vec<TreeNode<Node>>, depth: u32) -> Result<SearchTree, EngineError> {
        info!("Engine started exploring...");
        let mut result = Arc::new(Mutex::new(SearchTree {
            levels: Vec::new(),
            query: query.to_string(),
            depth,
        }));
        let q_embed = self.embed(query, EMBEDDING_PAD_LEN);
        let tasks: Vec<_> = nodes.into_iter()
            .map(|root| {
                let mut not_shared_self = self.clone();
                let res = result.clone();
                let q = q_embed.clone();
                let mut explored = Arc::clone(&self.explored);
                spawn(async move {
                    debug!("Spawned a new task for a node.");
                    not_shared_self.explore_node(&q, root, depth, res, explored).await;
                })
        })
        .collect();
        for task in tasks {
            task.await;
        }
        let unwrapped_result =Arc::try_unwrap(result)
            .map_err(|err| EngineError::SearchTreeUnwrapError(String::from("Failed to unwrap result")))?
            .into_inner();

        Ok(unwrapped_result)
    }

    async fn explore_node(&self, q_emb: &Option<Vec<f32>>, tree: TreeNode<Node>, depth: u32, result: Arc<Mutex<SearchTree>>, explored: Self::ArcEx) {
        // Check if the node has already been explored
        if explored.contains(&tree.node.url) {
            info!("Node is already explored. Skipped. | Url: {}.", &tree.node.url);
            return; // Skip this node
        }
        // Mark the node as explored
        //explored.insert(tree.node.url.clone());
        let mut level_counter = 0;
        let mut current_nodes = vec![tree.clone()];

        while level_counter < depth {
            info!("Exploring node. | Node:{}. | Depth: {}/{}.", &tree.node.url, level_counter+1, &depth);

            let mut next_nodes = vec![];

            for link_node in &mut current_nodes {
                if link_node.node.explored || explored.contains(&link_node.node.url) {
                    info!("Link has been explored previously. Skipped. | Url: {}.", &link_node.node.url);
                    continue;
                }

                self.fetch_children(q_emb, link_node, depth - level_counter).await;
                next_nodes.extend(link_node.children.clone());
            }

            result.lock().await.levels.push(DepthLevel { level: level_counter, links: current_nodes.clone() });

            level_counter += 1;
            current_nodes = next_nodes;
        }
    }

    async fn score_link(&self, q_emb: &Option<Vec<f32>>, link: &str) -> Result<f64, Self::E> {
        // Extract `content` from response json and use it to dismiss irrelavant links
        let json = self.engine.request(link, Some("json")).await?
            .json::<Value>()
            .await
            .unwrap();

        let content = json.get("results").and_then(|v| 
            v.as_array().and_then(|arr| arr.get(0).and_then(|c| c.get("content").and_then(|c| c.as_str()))).map(String::from)
        ).unwrap_or(String::from(""));

        //let len = query.split_whitespace().count();
        let c_emb = self.embed(&content, EMBEDDING_PAD_LEN);

        let score = Self::cosine_similarity(q_emb, &c_emb);
        Ok(score)
    }
}



