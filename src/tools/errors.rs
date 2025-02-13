

use thiserror::Error;


#[derive(Debug, Error)]
pub enum ToolError {
    
}


#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Error 403: Access denied. Ensure the `format` key is set in settings.yml of the searxng container and the value you are using is added: {0}")]
    QueryError(reqwest::Error, String),
    
    #[error("Request failed: {0}")]
    RequestError(reqwest::Error),

    #[error("Format error: {0}")]
    FormatError(String),

    #[error("Exploration error: {0}")]
    SearchTreeUnwrapError(String),
}