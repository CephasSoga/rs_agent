#![allow(unused)]

use std::sync::Arc;
use std::collections::HashMap;

use reqwest::{Client, Response};
use scraper::{Html, Selector};
use serde_json::json;
use tokio::{spawn, join};
use tokio::sync::{Mutex, mpsc};
use futures_util;
use tracing::{debug, info};

use crate::tools::errors::EngineError;

type LinkMap = HashMap<String, String>;
type OptionalURLs = Vec<Option<String>>;

const CLIENT_CONN_POOL_SIZE: usize = 12;
const CLIENT_CONN_TIMEOUT: u64 = 30;


pub trait HtmlParser {
    type E; // The error
    async fn extract(&self, html: &str, selector: &str) -> Vec<String>;
    async fn extract_tag_from_response(&self, response: Response, selector: &str) -> String;

}
pub trait RequestHandler {
    type E; // The error
    type Conc; // The concurrently accessed map of responses 
    async fn request(&self, query: &str, format: Option<&str>) -> Result<Response, EngineError>;
    async fn request_concurrent(&self, query: String, format: &[String]) -> Result<Self::Conc, Self::E>;
    async fn collect_urls(&self, query: &str, format: Option<&str>) -> Result<OptionalURLs, EngineError>;
    async fn search(&self, query: &str, format: Option<&str>, selector: &str) -> Result<LinkMap, EngineError>;
}

#[derive(Debug, Clone)]
pub struct MetaSearchEngine{
    pub host: String,
    pub client: Client,
}

impl MetaSearchEngine {
    pub fn new(host: &str) -> Self {
        Self{
            host: host.to_string(),
            client: Client::builder()
                .pool_max_idle_per_host(CLIENT_CONN_POOL_SIZE) // Maximum idle connections per host
                .timeout(std::time::Duration::from_secs(CLIENT_CONN_TIMEOUT)) // Set a timeout
                .build()
                .unwrap(),
        }
    }

    fn clone(&self) -> Self {
        Self {
            host: self.host.clone(),
            client: self.client.clone(),
        }
    }
}
impl HtmlParser for MetaSearchEngine {
    type E = EngineError;
    async fn extract(&self, html: &str, selector_str: &str) -> Vec<String> {
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

    async fn extract_tag_from_response(&self, response: Response, selector: &str) -> String {
        let html = response.text().await.unwrap();
        self.extract(&html, selector).await.join(" ")
    }
}

impl RequestHandler for MetaSearchEngine {
    type E = EngineError;
    type Conc = HashMap<String, Response>;
    async fn request(&self, query: &str, format: Option<&str>) -> Result<Response, EngineError> {
        debug!("Started request: <non-concurrent>. | Query: {}.", &query);
        let format = format.unwrap_or("json");
        if !vec!["json", "html", "csv", "rss"].contains(&format) {
            return Err(EngineError::FormatError(format!("Your request failed because of an invalid output format: `{} `.
                Valid options are ['json', 'html', csv', 'rss']. Use `None` if you are not sure; it will safely default to json.", format)));
        }

        let params = json!({
            "q": query,
            //"p": 1,
            "format": format,
        });

        let response = self.client.get(&self.host)
            .query(&params)
            .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|err| EngineError::RequestError(err))?
            .error_for_status()
            .map_err(|err| {
                if let Some(status) = err.status() {
                    let code = status.as_u16();
                    match code {
                        403 => EngineError::QueryError(err,
                        format!("Ensure the `format` key is set in settings.yml of the searxng container and the value `{}` added to it. 
                            Valid options are ['json', 'html', csv', 'rss']", format)),
                        _ => EngineError::RequestError(err),
                    }
                } else {
                    EngineError::RequestError(err)
                }
            });
        
        response
    }

    async fn request_concurrent(&self, query: String, format: &[String]) -> Result<Self::Conc, Self::E> {
        debug!("Started concurrent requests. | Query: {}.", &query);
        let (tx, mut rx) = mpsc::channel(format.len()+1);
        let mut responses = HashMap::new();

        let formats: Vec<String> = format.iter().map(|f| f.to_string()).collect();
        let mut tasks = vec![];
        let semaphore = Arc::new(tokio::sync::Semaphore::new(format.len()));

        for f in formats {
            let not_shared_self = self.clone();
            let q = query.clone();
            let f = f.clone();
            let tx_clone = tx.clone();
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let task = spawn(async move {
                debug!("Sending request for format: {}...", f);
                let r = not_shared_self.request(&q, Some(&f)).await;
                debug!("Received response for format: {}", f);
                if let Ok(response) = r {
                    let _ = tx_clone.send((f.clone(), response)).await;
                    debug!("Finished for format: {}", &f);
                }
                drop(permit);
            });
            tasks.push(task);
        }

        let r = futures_util::future::join_all(tasks).await;
        drop(tx);
        
        let mut count = 0; 
        while let Some((key, response)) = rx.recv().await {
            if count == format.len() {
                debug!("Finished concurrent requests. Breaking...");
                break;
            }
            responses.insert(key.clone(), response);
            count += 1;
            debug!("Inserted into response map with key: {}", &key);
        }
        debug!("Map: {:#?}", &responses);
        Ok(responses)
    }

    async fn collect_urls(&self, query: &str, format: Option<&str>) -> Result<OptionalURLs, EngineError> {
        let mut urls = vec![];
        
        let response = self.request(query, format).await?;
        let json  = response.json::<serde_json::Value>().await.unwrap_or_else(|err| {
            serde_json::Value::Null
        });

        if let Some(results) = json.get("results") {
            for result in results.as_array().unwrap() {
                let url = result.get("url").and_then(|url| url.as_str()).map(String::from);
                urls.push(url);
            }
        }

        Ok(urls)

    }

    async fn search(&self, query: &str, format: Option<&str>, selector: &str) -> Result<LinkMap, EngineError> {
        let mut res: LinkMap = LinkMap::new();

        let urls = self.collect_urls(query, format).await?;

        for url in urls {
            if let Some(url) = url {
                let response = self.client.get(&url).send().await.unwrap();
                let text = self.extract_tag_from_response(response, selector).await;
                res.insert(url, text);
            }
        }
        Ok(res)
    }
}