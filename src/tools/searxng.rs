#![allow(unused)]

use std::collections::HashMap;

use reqwest::{Client, Response};
use scraper::{Html, Selector};
use serde_json::json;
use tokio::join;

use crate::tools::errors::EngineError;

type LinkMap = HashMap<String, String>;
type OptionalURLs = Vec<Option<String>>;

const CLIENT_CONN_POOL_SIZE: usize = 5;
const CLIENT_CONN_TIMEOUT: u64 = 10;


pub trait HtmlParser {
    type E; // The error
    async fn extract(&self, html: &str, selector: &str) -> Vec<String>;
    async fn extract_tag_from_response(&self, response: Response, selector: &str) -> String;

}
pub trait RequestHandler {
    type E; // The error
    async fn request(&self, query: &str, format: Option<&str>) -> Result<Response, EngineError>;
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
    async fn request(&self, query: &str, format: Option<&str>) -> Result<Response, EngineError> {
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