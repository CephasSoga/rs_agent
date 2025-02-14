use tracing::info;
use std::time::Instant;
use crate::looging::setup_logger;

#[macro_export]
macro_rules! time {
    ($code:block) => {{
        let start = Instant::now();
        let result = $code; // Execute the code block
        let duration = start.elapsed();
        info!("Execution time: {:?}", duration);
        result // Return the result of the code block
    }};
}