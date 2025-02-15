use tracing::info;
use std::time::Instant;
use crate::looging::setup_logger;

#[macro_export]
macro_rules! time {
    ($label:expr, $code:block) => {{
        let start = Instant::now();
        let result = $code; // Execute the code block
        let duration = start.elapsed();
        info!("Execution time for {}: {:?}", $label, duration);
        result // Return the result of the code block
    }};
}